#!/usr/bin/env python3
"""Offline selective early LoRA training from teacher cache records."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from diffusers import FluxKontextPipeline
from peft import LoraConfig
from early_edit_reward_distillation.cache import load_teacher_record
from early_edit_reward_distillation.lora import velocity_diagnostics
from early_edit_reward_distillation.trajectory import _schedule

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--cache',required=True); p.add_argument('--output',required=True); p.add_argument('--steps',type=int,default=250); p.add_argument('--save-every',type=int,default=50); p.add_argument('--lr',type=float,default=5e-5); p.add_argument('--device',default='cuda'); args=p.parse_args()
    device=torch.device(args.device); cache=Path(args.cache); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    pipe=FluxKontextPipeline.from_pretrained(args.model,torch_dtype=torch.bfloat16,local_files_only=True).to(device); transformer=pipe.transformer
    transformer.set_attention_backend('native')
    for param in transformer.parameters(): param.requires_grad_(False)
    config=LoraConfig(r=4,lora_alpha=4,lora_dropout=0.0,bias='none',target_modules=['to_q','to_k','to_v','to_out.0'])
    transformer.add_adapter(config)
    trainable=[param for param in transformer.parameters() if param.requires_grad]
    if not trainable: raise RuntimeError('transformer.add_adapter created no trainable parameters')
    optimizer=torch.optim.AdamW(trainable,lr=args.lr)
    records=[]
    for folder in sorted(cache.iterdir()):
        if folder.is_dir() and (folder/'metadata.json').exists(): records.append(load_teacher_record(cache,folder.name))
    if not records: raise RuntimeError('teacher cache is empty')
    losses=[]; step=0
    while step < args.steps:
        tensors,meta=records[step % len(records)]; indices=list(meta['teacher_step_indices']); generated=int(meta['state_metadata']['generated_tokens']);
        timesteps,_mu=_schedule(pipe,int(meta['state_metadata']['steps']),device,generated)
        idx=indices[step % len(indices)]; lat=tensors[f'winner_state_t{step % len(indices)}'].unsqueeze(0).to(device); image_lat=tensors['image_latents'].unsqueeze(0).to(device); image_ids=tensors['image_ids'].to(device); prompt=tensors['prompt_embeds'].unsqueeze(0).to(device); pooled=tensors['pooled_prompt_embeds'].unsqueeze(0).to(device); text_ids=tensors['text_ids'].to(device); target=tensors[f'teacher_velocity_t{step % len(indices)}'].unsqueeze(0).to(device); mask=tensors['token_mask'].unsqueeze(0).to(device)
        output=transformer(hidden_states=torch.cat([lat,image_lat],dim=1),timestep=timesteps[idx].expand(1).to(device,dtype=lat.dtype)/1000,guidance=torch.full((1,),float(meta['state_metadata']['guidance_scale']),device=device),pooled_projections=pooled,encoder_hidden_states=prompt,txt_ids=text_ids,img_ids=image_ids,joint_attention_kwargs={},return_dict=False)[0][:,:generated]
        m=mask.to(output.dtype).unsqueeze(-1); loss=((output-target).square()*m).sum()/m.sum().clamp_min(1.0); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.item())); step+=1
        if step % args.save_every == 0 or step == args.steps:
            torch.save({'step':step,'loss':losses[-1],'loss_history':losses,'state_dict':{k:v.detach().cpu() for k,v in transformer.state_dict().items() if 'lora_' in k}},out/f'adapter_step_{step:04d}.pt')
            (out/'training.json').write_text(json.dumps({'steps':step,'learning_rate':args.lr,'rank':4,'dropout':0.0,'teacher_indices':[1,2],'loss_history':losses})+'\n')
    print(json.dumps({'steps':step,'final_loss':losses[-1],'output':str(out)}))

if __name__=='__main__': main()
