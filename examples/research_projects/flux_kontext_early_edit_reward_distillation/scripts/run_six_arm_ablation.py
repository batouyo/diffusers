#!/usr/bin/env python3
"""Five-sample, six-arm ablation around the existing selective LoRA checkpoint."""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw
from diffusers import FluxKontextPipeline
from peft import LoraConfig
from early_edit_reward_distillation.core import critical_nonzero_steps, native_euler_sde_step
from early_edit_reward_distillation.metrics import region_l1
from early_edit_reward_distillation.trajectory import prepare_state, _sigmas

ARMS = {
    "B0_fixed_selective_lora": dict(search=False, reward=False, coupled=False, dynamic=False),
    "B1_lora_early_search": dict(search=True, reward=False, coupled=False, dynamic=False),
    "B2_lora_search_reward": dict(search=True, reward=True, coupled=False, dynamic=False),
    "B3_lora_coupled_noise": dict(search=True, reward=False, coupled=True, dynamic=False),
    "B4_dynamic_selective_lora": dict(search=False, reward=False, coupled=False, dynamic=True),
    "B5_all_methods": dict(search=True, reward=True, coupled=True, dynamic=True),
}

@torch.inference_mode()
def decode(pipe, state, latents):
    u = pipe._unpack_latents(latents, state.height, state.width, pipe.vae_scale_factor)
    u = u / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    return pipe.image_processor.postprocess(pipe.vae.decode(u.to(pipe.vae.dtype), return_dict=False)[0], output_type="pil")

def load_lora(pipe, checkpoint):
    tr = pipe.transformer
    tr.add_adapter(LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0, bias="none", target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    tr.load_state_dict(ck["state_dict"], strict=False)
    tr.set_adapters(["default"], [1.0])

def transformer_call(pipe, state, latents, timestep):
    b = latents.shape[0]
    return pipe.transformer(hidden_states=torch.cat([latents, state.image_latents.repeat(b, 1, 1)], 1), timestep=timestep.expand(b).to(latents.dtype)/1000, guidance=torch.full((b,), float(state.metadata["guidance_scale"]), device=latents.device), pooled_projections=state.pooled_prompt_embeds.repeat(b, 1), encoder_hidden_states=state.prompt_embeds.repeat(b, 1, 1), txt_ids=state.text_ids, img_ids=state.image_ids, joint_attention_kwargs={}, return_dict=False)[0][:, :latents.shape[1]]

@torch.inference_mode()
def velocity(pipe, state, latents, timestep, arm, token_mask, key_indices):
    idx = int(pipe.scheduler.index_for_timestep(timestep))
    active = (not arm["dynamic"]) or idx in key_indices
    tr = pipe.transformer
    if not active:
        tr.set_adapters(["default"], [0.0])
        return transformer_call(pipe, state, latents, timestep)
    tr.set_adapters(["default"], [1.0])
    adapted = transformer_call(pipe, state, latents, timestep)
    if not arm["dynamic"]:
        return adapted
    tr.set_adapters(["default"], [0.0])
    base = transformer_call(pipe, state, latents, timestep)
    tr.set_adapters(["default"], [1.0])
    m = token_mask.to(adapted.dtype).reshape(1, -1, 1)
    return base + (adapted - base) * m

@torch.inference_mode()
def rollout(pipe, state, latents, start, arm, token_mask, key_indices):
    cur = latents
    for i in range(start, len(state.timesteps)):
        pred = velocity(pipe, state, cur, state.timesteps[i], arm, token_mask, key_indices)
        sigma, nxt = _sigmas(pipe, state.timesteps[i])
        cur = (cur.float() + (nxt - sigma) * pred.float()).to(cur.dtype)
    return cur

@torch.inference_mode()
def rollout_until(pipe, state, latents, start, target, arm, token_mask, key_indices):
    cur = latents
    for i in range(start, target):
        pred = velocity(pipe, state, cur, state.timesteps[i], arm, token_mask, key_indices)
        sigma, nxt = _sigmas(pipe, state.timesteps[i])
        cur = (cur.float() + (nxt - sigma) * pred.float()).to(cur.dtype)
    return cur

def score_value(scorer, source, image, instruction, seed):
    old = getattr(scorer, "seed", None)
    if old is not None: scorer.seed = int(seed)
    value = scorer.evaluate([source, image], instruction)["overall"]
    if old is not None: scorer.seed = old
    return float(value.item() if hasattr(value, "item") else value)

def masked_lpips(metric, source, image, mask):
    if metric is None: return float("nan")
    device = next(metric.parameters()).device
    a = torch.from_numpy(np.asarray(source).astype("float32") / 127.5 - 1).permute(2,0,1).unsqueeze(0).to(device)
    b = torch.from_numpy(np.asarray(image).astype("float32") / 127.5 - 1).permute(2,0,1).unsqueeze(0).to(device)
    keep = (torch.from_numpy(np.asarray(mask).astype("float32"))/255).unsqueeze(0).repeat(3,1,1).unsqueeze(0).to(device)
    return float(metric(a * keep, b * keep).mean().item())

@torch.inference_mode()
def run_arm(pipe, scorer, metric, records, samples_root, checkpoint, arm_name, seed_base, output, device):
    arm = ARMS[arm_name]; arm_dir = output / arm_name; arm_dir.mkdir(parents=True, exist_ok=True); rows=[]
    for ri, rec in enumerate(records):
        sid=str(rec["sample_id"]); folder=Path(samples_root)/sid; source=Image.open(folder/"source.png").convert("RGB"); mask=Image.open(folder/"edit_mask.png").convert("L"); instruction=str(rec["instruction"]); seed=int(seed_base+ri*100)
        state=prepare_state(pipe,source,instruction,seed,height=512,width=512,steps=28,guidance_scale=3.5,device=device)
        th,tw=state.height//(pipe.vae_scale_factor*2),state.width//(pipe.vae_scale_factor*2); token_mask=torch.nn.functional.interpolate(torch.from_numpy(np.asarray(mask,dtype="float32"))[None,None],size=(th,tw),mode="area")[0,0].flatten().to(device)>.5
        key_indices={int(x["index"]) for x in critical_nonzero_steps(pipe.scheduler.sigmas.detach().cpu().flatten().tolist())[:2]}
        base=rollout(pipe,state,state.latents,0,arm,token_mask,key_indices); base_img=decode(pipe,state,base)[0]
        winner=base; search_records=[]
        if arm["search"]:
            cur=state.latents; cur_step=0
            for stage,idx in enumerate(sorted(key_indices)):
                cur=rollout_until(pipe,state,cur,cur_step,idx,arm,token_mask,key_indices)
                pred=velocity(pipe,state,cur,state.timesteps[idx],arm,token_mask,key_indices); sigma,nxt=_sigmas(pipe,state.timesteps[idx]); gen=torch.Generator(device=device).manual_seed(seed+10000+stage); shared=torch.randn(cur.shape,generator=gen,device=device); indep=torch.randn((4,)+tuple(cur.shape[1:]),generator=gen,device=device)
                noise=indep if not arm["coupled"] else indep*(~token_mask.reshape(1,-1,1)).to(indep.dtype)+shared.expand_as(indep)*token_mask.reshape(1,-1,1).to(indep.dtype)
                candidates=[]; terminals=[]
                for j in range(4):
                    cand,_=native_euler_sde_step(cur,pred,sigma,nxt,noise[j],alpha=.2,diffusion_scale=1.0,first_step=idx==0); candidates.append(cand); terminals.append(rollout(pipe,state,cand,idx+1,arm,token_mask,key_indices))
                images=[decode(pipe,state,x)[0] for x in terminals]
                rewards=[score_value(scorer,source,im,instruction,seed+stage*1000+j) for j,im in enumerate(images)] if arm["reward"] else [0.0,0.0,0.0,0.0]
                winner_idx=(max(range(4),key=lambda j:rewards[j]) if arm["reward"] else 0); search_records.append({"stage":stage,"step_index":idx,"rewards":rewards,"winner_index":winner_idx}); cur=candidates[winner_idx]; cur_step=idx+1; winner=terminals[winner_idx]
        final_img=decode(pipe,state,winner)[0]; reward=score_value(scorer,source,final_img,instruction,seed+90000); rows.append({"sample_id":sid,"arm":arm_name,"reward":reward,"edit_l1":region_l1(source,final_img,mask,preserve=False),"preserve_l1":region_l1(source,final_img,mask,preserve=True),"preserve_lpips":masked_lpips(metric,source,final_img,mask),"scale":1.0,"seed":seed,"resolution":"512x512","generated_tokens":state.metadata["generated_tokens"],"source_conditioning_tokens":state.metadata["source_conditioning_tokens"]})
        sample_dir=arm_dir/sid; sample_dir.mkdir(parents=True,exist_ok=True); source.save(sample_dir/"source.png"); mask.save(sample_dir/"edit_mask.png"); base_img.save(sample_dir/"baseline_internal.png"); final_img.save(sample_dir/"result.png"); (sample_dir/"search.json").write_text(json.dumps(search_records,indent=2)+"\n")
    with (arm_dir/"results.csv").open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--manifest",required=True); p.add_argument("--samples-root",required=True); p.add_argument("--output",required=True); p.add_argument("--editscore-model",required=True); p.add_argument("--editscore-lora",required=True); p.add_argument("--arms",required=True); p.add_argument("--count",type=int,default=5); p.add_argument("--seed",type=int,default=20260830); p.add_argument("--device",default="cuda"); a=p.parse_args(); device=torch.device(a.device); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); records=json.loads(Path(a.manifest).read_text())[:a.count]
    pipe=FluxKontextPipeline.from_pretrained(a.model,torch_dtype=torch.bfloat16,local_files_only=True).to(device); pipe.set_progress_bar_config(disable=True); load_lora(pipe,a.checkpoint); sys.path.insert(0,"/home/hyp/Code/EditScore"); from editscore import EditScore; scorer=EditScore(backbone="qwen3vl",model_name_or_path=a.editscore_model,lora_path=a.editscore_lora,score_range=25,num_pass=1)
    # LPIPS is intentionally disabled during the time-critical GPU run. Its
    # convolutional backbone is evaluated in a separate post-processing pass.
    metric = None
    all_rows=[]
    for name in a.arms.split(","): all_rows.extend(run_arm(pipe,scorer,metric,records,a.samples_root,a.checkpoint,name,a.seed,out,device))
    with (out/"ablation_results.csv").open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
    summary=[]
    for name in a.arms.split(","):
        x=[r for r in all_rows if r["arm"]==name]; summary.append({"arm":name,"n":len(x),"reward_mean":float(np.mean([float(r["reward"]) for r in x])),"reward_std":float(np.std([float(r["reward"]) for r in x])),"edit_l1_mean":float(np.mean([float(r["edit_l1"]) for r in x])),"preserve_l1_mean":float(np.mean([float(r["preserve_l1"]) for r in x])),"preserve_lpips_mean":float(np.nanmean([float(r["preserve_lpips"]) for r in x]))})
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
