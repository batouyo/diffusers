#!/usr/bin/env python3
"""Build velocity teacher records from fixed train samples.

This stage performs no reward evaluation.  The winner trajectory is generated
with a fixed, reproducible coupled candidate policy; metadata records that
selection source so it cannot be confused with an EditScore-selected path.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from diffusers import FluxKontextPipeline
from early_edit_reward_distillation.cache import save_teacher_record
from early_edit_reward_distillation.core import critical_nonzero_steps
from early_edit_reward_distillation.trajectory import branch_step, prepare_state, rollout_until, velocity

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--manifest',required=True); p.add_argument('--samples-root',required=True); p.add_argument('--output',required=True); p.add_argument('--seed',type=int,default=20260830); p.add_argument('--height',type=int,default=512); p.add_argument('--width',type=int,default=512); p.add_argument('--steps',type=int,default=28); p.add_argument('--guidance',type=float,default=3.5); p.add_argument('--alpha',type=float,default=.2); p.add_argument('--device',default='cuda'); args=p.parse_args()
    manifest=json.loads(Path(args.manifest).read_text()); device=torch.device(args.device)
    pipe=FluxKontextPipeline.from_pretrained(args.model,torch_dtype=torch.bfloat16,local_files_only=True).to(device); pipe.set_progress_bar_config(disable=True)
    for ri,rec in enumerate(manifest):
        sid=str(rec['sample_id']); folder=Path(args.samples_root)/sid; source=Image.open(folder/'source.png').convert('RGB'); mask=Image.open(folder/'edit_mask.png').convert('L'); seed=int(args.seed+ri*100)
        state=prepare_state(pipe,source,str(rec['instruction']),seed,height=args.height,width=args.width,steps=args.steps,guidance_scale=args.guidance,device=device)
        th,tw=state.height//(pipe.vae_scale_factor*2),state.width//(pipe.vae_scale_factor*2)
        token_mask=torch.nn.functional.interpolate(torch.from_numpy(np.asarray(mask,dtype='float32'))[None,None],size=(th,tw),mode='area')[0,0].flatten().to(device)>.5
        indices=[int(x['index']) for x in critical_nonzero_steps(pipe.scheduler.sigmas.detach().cpu().flatten().tolist())[:2]]
        base_states=[]; win_states=[]; base_vel=[]; win_vel=[]; current=state.latents; winner=state.latents; cur_step=0
        for stage,idx in enumerate(indices):
            current=rollout_until(pipe,state,current,cur_step,idx); winner=rollout_until(pipe,state,winner,cur_step,idx)
            base_states.append(current.detach().cpu()); base_vel.append(velocity(pipe,state,current,state.timesteps[idx]).detach().cpu())
            win_states.append(winner.detach().cpu()); win_vel.append(velocity(pipe,state,winner,state.timesteps[idx]).detach().cpu())
            candidates,_=branch_step(pipe,state,winner,idx,token_mask,seed+10000+stage,mode='native_euler_sde',alpha=args.alpha)
            winner=candidates[:1]; cur_step=idx+1
        tensors={}
        for i in range(2):
            tensors[f'baseline_state_t{i}']=base_states[i][0]; tensors[f'winner_state_t{i}']=win_states[i][0]; tensors[f'baseline_velocity_t{i}']=base_vel[i][0]; tensors[f'teacher_velocity_t{i}']=win_vel[i][0]; tensors[f'delta_velocity_t{i}']=win_vel[i][0]-base_vel[i][0]
        tensors['token_mask']=token_mask.cpu()
        tensors.update({'image_latents': state.image_latents[0].cpu(), 'image_ids': state.image_ids.cpu(), 'prompt_embeds': state.prompt_embeds[0].cpu(), 'pooled_prompt_embeds': state.pooled_prompt_embeds[0].cpu(), 'text_ids': state.text_ids.cpu()})
        meta={'instruction':str(rec['instruction']),'seed':seed,'teacher_step_indices':indices,'branch_mode':'native_euler_sde','alpha':args.alpha,'winner_selection':'fixed_coupled_candidate_zero_no_reward','reward_search_in_cache':False,'state_metadata':state.metadata,'mask_area':float(rec.get('mask_area',0.0))}
        save_teacher_record(args.output,sid,tensors,meta); print(json.dumps({'sample_id':sid,'indices':indices}))

if __name__=='__main__': main()
