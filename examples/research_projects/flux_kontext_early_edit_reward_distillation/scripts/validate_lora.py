import argparse, json
from pathlib import Path
import torch
from diffusers import FluxKontextPipeline
from early_edit_reward_distillation.cache import load_teacher_record
from early_edit_reward_distillation.trajectory import _schedule

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--cache',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--device',default='cuda'); a=p.parse_args(); d=torch.device(a.device)
 pipe=FluxKontextPipeline.from_pretrained(a.model,torch_dtype=torch.bfloat16,local_files_only=True).to(d); tr=pipe.transformer; tr.set_attention_backend('_native_math')
 for x in tr.parameters(): x.requires_grad_(False)
 from peft import LoraConfig
 tr.add_adapter(LoraConfig(r=4,lora_alpha=4,lora_dropout=0.,bias='none',target_modules=['to_q','to_k','to_v','to_out.0']))
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True); tr.load_state_dict(ck['state_dict'],strict=False)
 root=Path(a.cache); sid=next(x.name for x in root.iterdir() if x.is_dir()); t,m=load_teacher_record(root,sid); idx=int(m['teacher_step_indices'][0]); n=int(m['state_metadata']['generated_tokens']); ts,_=_schedule(pipe,int(m['state_metadata']['steps']),d,n)
 def forward():
  lat=t['winner_state_t0'].unsqueeze(0).to(d); il=t['image_latents'].unsqueeze(0).to(d); out=tr(hidden_states=torch.cat([lat,il],1),timestep=ts[idx].expand(1).to(d,dtype=lat.dtype)/1000,guidance=torch.full((1,),float(m['state_metadata']['guidance_scale']),device=d),pooled_projections=t['pooled_prompt_embeds'].unsqueeze(0).to(d),encoder_hidden_states=t['prompt_embeds'].unsqueeze(0).to(d),txt_ids=t['text_ids'].to(d),img_ids=t['image_ids'].to(d),joint_attention_kwargs={},return_dict=False)[0][:,:n]; return out
 pred=forward(); target=t['teacher_velocity_t0'].unsqueeze(0).to(d); mask=t['token_mask'].unsqueeze(0).to(d).to(pred.dtype).unsqueeze(-1); mse=((pred-target).square()*mask).sum()/mask.sum().clamp_min(1.)
 tr.disable_adapters(); disabled=forward(); tr.enable_adapters(); base_delta=(pred-disabled).abs().max(); tr.disable_adapters(); zero_delta=(disabled-forward()).abs().max()
 print(json.dumps({'sample_id':sid,'masked_mse':float(mse),'adapter_delta_max':float(base_delta),'disabled_repeat_max':float(zero_delta),'checkpoint':a.checkpoint}))
if __name__=='__main__': main()
