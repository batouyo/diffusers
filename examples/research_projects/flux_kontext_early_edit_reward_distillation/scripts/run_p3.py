#!/usr/bin/env python3
import argparse,csv,json,sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from diffusers import FluxKontextPipeline
from early_edit_reward_distillation.trajectory import prepare_state,deterministic_rollout,two_stage_search
from early_edit_reward_distillation.metrics import region_l1

@torch.inference_mode()
def decode(pipe,state,lat):
 u=pipe._unpack_latents(lat,state.height,state.width,pipe.vae_scale_factor); u=u/pipe.vae.config.scaling_factor+pipe.vae.config.shift_factor
 return pipe.image_processor.postprocess(pipe.vae.decode(u.to(pipe.vae.dtype),return_dict=False)[0],output_type='pil')
def val(x): return float(x.item() if hasattr(x,'item') else x)
def load_adapter(pipe,path,scale):
 from peft import LoraConfig
 tr=pipe.transformer; tr.add_adapter(LoraConfig(r=4,lora_alpha=4,lora_dropout=0.,bias='none',target_modules=['to_q','to_k','to_v','to_out.0']))
 ck=torch.load(path,map_location='cpu',weights_only=True); tr.load_state_dict(ck['state_dict'],strict=False)
 try: tr.set_adapters(['default'],[float(scale)])
 except Exception: pass
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--manifest',required=True); p.add_argument('--samples-root',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True); p.add_argument('--editscore-model',required=True); p.add_argument('--editscore-lora',required=True); p.add_argument('--start',type=int,default=0); p.add_argument('--end',type=int,default=8); p.add_argument('--scale',type=float,default=1.0); p.add_argument('--seed',type=int,default=20260830); p.add_argument('--device',default='cuda')
 a=p.parse_args(); d=torch.device(a.device); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); man=json.loads(Path(a.manifest).read_text()); pipe=FluxKontextPipeline.from_pretrained(a.model,torch_dtype=torch.bfloat16,local_files_only=True).to(d); pipe.set_progress_bar_config(disable=True); load_adapter(pipe,a.checkpoint,a.scale)
 sys.path.insert(0,'/home/hyp/Code/EditScore'); from editscore import EditScore; scorer=EditScore(backbone='qwen3vl',model_name_or_path=a.editscore_model,lora_path=a.editscore_lora,score_range=25,num_pass=1)
 rows=[]
 for ri,rec in enumerate(man[a.start:a.end],a.start):
  sid=str(rec['sample_id']); folder=Path(a.samples_root)/sid; source=Image.open(folder/'source.png').convert('RGB'); mask=Image.open(folder/'edit_mask.png').convert('L'); seed=a.seed+ri*100
  state=prepare_state(pipe,source,str(rec['instruction']),seed,height=512,width=512,steps=28,guidance_scale=3.5,device=d); th,tw=state.height//(pipe.vae_scale_factor*2),state.width//(pipe.vae_scale_factor*2); tm=torch.nn.functional.interpolate(torch.from_numpy(np.asarray(mask,dtype='float32'))[None,None],size=(th,tw),mode='area')[0,0].flatten().to(d)>.5
  def score(ls):
   ims=decode(pipe,state,ls); return [val(scorer.evaluate([source,x],str(rec['instruction']))['overall']) for x in ims]
  base=deterministic_rollout(pipe,state,state.latents,0); baseim=decode(pipe,state,base)[0]
  for method,im in [('E_lora_baseline',baseim)]: rows.append({'sample_id':sid,'scale':a.scale,'method':method,'reward':val(scorer.evaluate([source,im],str(rec['instruction']))['overall']),'preserve_l1':region_l1(source,im,mask,preserve=True)})
  term,recs=two_stage_search(pipe,state,tm,score,seed=seed+10000,alpha=.2,baseline_terminal=base); im=decode(pipe,state,term)[0]; rows.append({'sample_id':sid,'scale':a.scale,'method':'F_coupled_lora','reward':val(scorer.evaluate([source,im],str(rec['instruction']))['overall']),'preserve_l1':region_l1(source,im,mask,preserve=True)}); (out/f'{sid}_search.json').write_text(json.dumps(recs,indent=2,default=str)); print(json.dumps({'sample_id':sid,'scale':a.scale}),flush=True)
 with (out/f'results_{a.start}_{a.end}_{a.scale}.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
if __name__=='__main__': main()
