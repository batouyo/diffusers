#!/usr/bin/env python3
"""Paired four-arm FLUX-Kontext continuous-strength reward ablation."""
from __future__ import annotations
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np, torch
from PIL import Image,ImageDraw
from diffusers import FluxKontextPipeline
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from early_edit_reward_distillation.continuous_strength import ContinuousStrengthConfig,TrajectoryTrace,estimate_edit_token_mask,rollout_strengths,reference_velocity
from early_edit_reward_distillation.core import coupled_noise,critical_nonzero_steps,native_euler_sde_step,tensor_hash
from early_edit_reward_distillation.metrics import region_l1
from early_edit_reward_distillation.rewards import build_official_editscore
from early_edit_reward_distillation.trajectory import _sigmas,deterministic_rollout,prepare_state,velocity
ARMS={'velo_baseline': {'search': False, 'reward': False, 'coupled': False}, 'early_search': {'search': True, 'reward': False, 'coupled': False}, 'search_reward': {'search': True, 'reward': True, 'coupled': False}, 'independent_sde': {'search': True, 'reward': True, 'coupled': False, 'independent_all': True}, 'full': {'search': True, 'reward': True, 'coupled': True}}

@torch.inference_mode()
def decode(pipe,state,x):
    u=pipe._unpack_latents(x,state.height,state.width,pipe.vae_scale_factor)
    u=u/pipe.vae.config.scaling_factor+pipe.vae.config.shift_factor
    return pipe.image_processor.postprocess(pipe.vae.decode(u.to(pipe.vae.dtype),return_dict=False)[0],output_type='pil')[0]

def step(pipe,state,x,t,v):
    a,b=_sigmas(pipe,t)
    return (x.float()+(b-a)*v.float()).to(x.dtype)

def trace(prompt,states,vels,times,sigmas,residuals):
    return TrajectoryTrace(prompt,states,vels,times,sigmas,states[-1].clone(),residuals)

@torch.inference_mode()
def deterministic_pair(pipe,pstate,estate):
    p,e=pstate.latents.clone(),estate.latents.clone();ps,es=[p.clone()],[e.clone()]
    pvs,evs,ts,ss,prs,ers=[],[],[],[],[],[]
    for t in estate.timesteps:
        vp,ve=velocity(pipe,pstate,p,t),velocity(pipe,estate,e,t);a,b=_sigmas(pipe,t)
        p=step(pipe,pstate,p,t,vp);e=step(pipe,estate,e,t,ve)
        pvs.append(vp.clone());evs.append(ve.clone());ts.append(float(t));ss.append((a,b));prs.append(torch.zeros_like(p));ers.append(torch.zeros_like(e));ps.append(p.clone());es.append(e.clone())
    return trace('preserve',ps,pvs,ts,ss,prs),trace('edit',es,evs,ts,ss,ers)

@torch.inference_mode()
def rollout_coupled(pipe,pstate,estate,mask,arm,scorer,source,instruction,seed,critical,alpha):
    p,e=pstate.latents.clone(),estate.latents.clone();ps,es=[p.clone()],[e.clone()]
    pvs,evs,ts,ss,prs,ers=[],[],[],[],[],[];records=[];branch_images=[];winner=0
    m=mask.to(e.device,dtype=torch.bool).reshape(1,-1,1); crit=set(critical)
    for i,t in enumerate(estate.timesteps):
        ve=velocity(pipe,estate,e,t);a,b=_sigmas(pipe,t);vp=reference_velocity(e,estate.image_latents,a);similarity=((vp.float().abs()+1e-8)/(vp.float().abs()+1e-8+(ve.float()-vp.float()).abs())).mean(dim=-1);vout=torch.where((similarity>=0.8).unsqueeze(-1),vp,ve).to(ve.dtype)
        vout = vout if i in crit else ve
        pm,em=step(pipe,pstate,p,t,vp),step(pipe,estate,e,t,vout);pn,en=pm,em;pr,er=torch.zeros_like(p),torch.zeros_like(e)
        if i in crit and (arm['coupled'] or arm['search']):
            g=torch.Generator(device=e.device).manual_seed(int(seed+i));shared=torch.randn(e.shape,generator=g,device=e.device,dtype=torch.float32)
            if arm['search']:
                independent=torch.randn((4,)+tuple(e.shape[1:]),generator=g,device=e.device,dtype=torch.float32)
                noises=coupled_noise(shared.expand_as(independent),independent,(~(similarity>=0.8)),rho=0.0) if arm['coupled'] and not arm.get('independent_all',False) else independent
                candidates=[]
                if arm['coupled'] or arm.get('independent_all',False): pn,_=native_euler_sde_step(p,vp,a,b,shared if arm['coupled'] else independent[0],alpha=alpha,first_step=i==0)
                terminals=[]
                for j in range(4):
                    c,_=native_euler_sde_step(e,vout,a,b,noises[j],alpha=alpha,first_step=i==0);candidates.append(c);terminals.append(deterministic_rollout(pipe,estate,c,i+1))
                imgs=[decode(pipe,estate,x) for x in terminals];branch_images.append(imgs)
                details=[scorer.score_details(source,x,instruction) for x in imgs] if arm['reward'] else []
                rewards=[float(d['overall']) for d in details] if details else [float('nan')]*4
                winner=max(range(4),key=lambda j:(rewards[j],-j)) if arm['reward'] else 0;en=candidates[winner]
                if arm['coupled']: pr=pn-pm
                er=en-em
                records.append({'step_index':i,'seed':int(seed+i),'winner_index':winner,'rewards':rewards,'reward_details':details,'state_hash':tensor_hash(e),'reward_residual_norm':float(er.float().norm()),'preserve_residual_norm':float(pr.float().norm()),'finite':bool(torch.isfinite(en).all())})
            else:
                pn,_=native_euler_sde_step(p,vp,a,b,shared if arm['coupled'] else torch.randn_like(shared),alpha=alpha,first_step=i==0)
                noise=coupled_noise(shared,torch.randn(e.shape,generator=g,device=e.device,dtype=torch.float32),(~(similarity>=0.8)),rho=0.0)
                en,_=native_euler_sde_step(e,vout,a,b,noise,alpha=alpha,first_step=i==0);pr=pn-pm;er=en-em
                records.append({'step_index':i,'seed':int(seed+i),'winner_index':0,'rewards':[],'reward_details':[],'state_hash':tensor_hash(e),'reward_residual_norm':float(er.float().norm()),'preserve_residual_norm':float(pr.float().norm()),'finite':bool(torch.isfinite(en).all())})
        pvs.append(vp.clone());evs.append(ve.clone());ts.append(float(t));ss.append((a,b));prs.append(pr.clone());ers.append(er.clone());p,e=pn,en;ps.append(p.clone());es.append(e.clone())
    return trace('preserve',ps,pvs,ts,ss,prs),trace('edit',es,evs,ts,ss,ers),records,winner,branch_images

def make_sheet(images,path):
    cell=192;out=Image.new('RGB',(cell*len(images),cell+24),'white');draw=ImageDraw.Draw(out)
    for i,(im,label) in enumerate(images):out.paste(im.convert('RGB').resize((cell,cell)),(i*cell,0));draw.text((i*cell+4,cell+4),label,fill='black')
    out.save(path)

def load_records(paths,count):
    out=[]
    for path in paths:out.extend(json.loads(Path(path).read_text(encoding='utf-8')))
    return out[:count]

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--manifest',required=True,nargs='+');p.add_argument('--samples-root',required=True);p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=5);p.add_argument('--seed',type=int,default=20260830);p.add_argument('--alpha',type=float,default=.05);p.add_argument('--critical-step-indices',default=None);p.add_argument('--editscore-model',default='/data15/hyp/weight/Qwen3-VL-4B-Instruct');p.add_argument('--editscore-lora',default='/data15/hyp/weight/EditScore-Qwen3-VL-4B-Instruct');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);records=load_records(a.manifest,a.count);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipe=FluxKontextPipeline.from_pretrained(a.model,torch_dtype=torch.bfloat16,local_files_only=True).to(dev);pipe.set_progress_bar_config(disable=True);scorer=build_official_editscore(a.editscore_model,a.editscore_lora,num_pass=1)
    strengths=tuple(float(x) for x in np.linspace(0.0,1.0,10));cfg=ContinuousStrengthConfig(alpha=a.alpha,steps=30,guidance_scale=2.5,intervention_step_count=4,similarity_threshold=0.8,critical_step_indices=None if a.critical_step_indices is None else tuple(int(x) for x in a.critical_step_indices.split(',')),strengths=strengths);allrows=[];critical=None
    for ri,record in enumerate(records):
        sid=str(record['sample_id']);sample_dir=Path(a.samples_root)/sid;source=Image.open(sample_dir/'source.png').convert('RGB');pixel_mask=Image.open(sample_dir/'edit_mask.png').convert('L');instruction=str(record['instruction']);seed=int(a.seed+ri*100)
        pstate=prepare_state(pipe,source,cfg.neutral_prompt,seed,height=512,width=512,steps=30,guidance_scale=2.5,device=dev);estate=prepare_state(pipe,source,instruction,seed,height=512,width=512,steps=30,guidance_scale=2.5,device=dev);pstate.latents=estate.latents.clone()
        pilot_p,pilot_e=deterministic_pair(pipe,pstate,estate)
        if critical is None:critical=list(cfg.critical_step_indices) if cfg.critical_step_indices is not None else list(critical_nonzero_steps([float(x) for x in pipe.scheduler.sigmas.detach().cpu().flatten().tolist()])[:cfg.critical_steps][i]["index"] for i in range(cfg.critical_steps))
        token_mask,mask_scores=estimate_edit_token_mask(pilot_p,pilot_e,critical,quantile=cfg.mask_quantile,min_ratio=cfg.min_edit_ratio,max_ratio=cfg.max_edit_ratio)
        for name,arm in ARMS.items():
            if arm['search'] or arm['coupled']:pt,et,branch_records,winner_index,branch_images=rollout_coupled(pipe,pstate,estate,token_mask,arm,scorer,source,instruction,seed+10000,critical,a.alpha)
            else:pt,et=pilot_p,pilot_e;branch_records=[];winner_index=None;branch_images=[]
            values=rollout_strengths(pipe,pt,et,strengths,preservation_state=pstate,edited_state=estate,source_latent=estate.image_latents,intervention_step_count=cfg.intervention_step_count,search_step_indices=critical,similarity_threshold=cfg.similarity_threshold,similarity_mode=cfg.similarity_mode);method_dir=out/name/sid;method_dir.mkdir(parents=True,exist_ok=True);source.save(method_dir/'source.png');pixel_mask.save(method_dir/'edit_mask.png');rendered=[]
            for strength,latent in values.items():
                image=decode(pipe,estate,latent);image.save(method_dir/f'strength_{strength:.2f}.png');rendered.append((image,f's={strength:.2f}'));allrows.append({'sample_id':sid,'method':name,'strength':strength,'edit_l1':region_l1(source,image,pixel_mask,False),'preserve_l1':region_l1(source,image,pixel_mask,True),'latent_norm':float(latent.float().norm()),'seed':seed,'winner_index':'' if winner_index is None else winner_index})
            make_sheet(rendered,method_dir/'contact_sheet.png')
            branch_dir=method_dir/'branch_candidates';branch_dir.mkdir(exist_ok=True)
            for stage,images in enumerate(branch_images,1):
                for j,image in enumerate(images):image.save(branch_dir/f'stage_{stage}_branch_{j}.png')
            payload={'sample_id':sid,'method':name,'seed':seed,'instruction':instruction,'critical_step_indices':critical,'generated_tokens':estate.metadata['generated_tokens'],'source_conditioning_tokens':estate.metadata['source_conditioning_tokens'],'token_mask':token_mask.cpu().tolist(),'mask_scores':mask_scores.cpu().tolist(),'winner_index':winner_index,'reward_used':arm['reward'],'branch_records':branch_records,'strengths':list(strengths)}
            (method_dir/'trajectory.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8');(method_dir/'branch_scores.json').write_text(json.dumps(branch_records,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    with (out/'paired_results.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    (out/'config.json').write_text(json.dumps({'methods':ARMS,'sample_count':len(records),'seed':a.seed,'strengths':list(strengths),'reward':'official EditScore Qwen3-VL-4B-Instruct','model':a.model},indent=2)+'\n',encoding='utf-8')
    with (out/'quantitative_summary.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=['method','strength','edit_l1_mean','preserve_l1_mean']);w.writeheader()
        for name in ARMS:
            for strength in strengths:
                rows=[r for r in allrows if r['method']==name and float(r['strength'])==strength];w.writerow({'method':name,'strength':strength,'edit_l1_mean':float(np.mean([r['edit_l1'] for r in rows])),'preserve_l1_mean':float(np.mean([r['preserve_l1'] for r in rows]))})
    (out/'SUMMARY.md').write_text(f'# Core ablation summary\n\nSamples: {len(records)}\nReward: official EditScore, num_pass=1\n\nThis is a paired 10-sample qualitative/quantitative ablation; no statistical significance is claimed.\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'samples':len(records),'rows':len(allrows),'critical_step_indices':critical},indent=2))
if __name__=='__main__':main()
