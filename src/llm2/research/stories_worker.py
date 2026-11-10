"""Persistent language worker: train-only LR calibration, fresh fit, text, profiles."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from ..checkpoint import read_checkpoint_metadata
from ..config import ExperimentConfig,load_config
from ..data import write_json
from ..trainer import Trainer
from ..runtime import attention_context,source_provenance,environment
from .stories_data import StoriesData,StoryStream,evaluation_batches
from .stories_protocol import configuration,PROTOCOL_ID,RATES,PROMPTS
from .binding_worker import training_benchmark
from .recall_worker import sync,state_storage_bytes
from .resource_telemetry import EnergyMeter,energy_result,host_memory
from .state_readout_worker import initial_digest,retain_final
from .state_study import materialize


@torch.no_grad()
def score(model,data,split,length,batch,device):
    was=model.training;model.eval();nll=0.;count=windows=batches=0;started=time.monotonic()
    try:
        with attention_context(device.type=='cpu'):
            for pack in evaluation_batches(data,split,length,batch,device):
                logits=model(pack['input_ids']).logits;targets=pack['targets'];valid=targets!=-100
                nll+=float(torch.nn.functional.cross_entropy(logits.float().flatten(0,1),targets.flatten(),ignore_index=-100,reduction='sum'))
                count+=int(valid.sum());windows+=len(targets);batches+=1
        sync(device)
    finally:model.train(was)
    if count!=data.manifest['splits'][split]['bytes']:raise ValueError('development coverage differs from full document bytes')
    return {'split':split,'nll':nll/count,'bits_per_byte':nll/count/math.log(2),'byte_perplexity':math.exp(min(50,nll/count)),
        'bytes':count,'windows':windows,'batches':batches,'documents':len(data.docs[split]),'wall_seconds':time.monotonic()-started,
        'scope':'every selected document byte once, tails included, BOS/padding excluded, reset per window'}


class StoryTrainer(Trainer):
    def __init__(self,cfg,*,calibration=False,smoke=False,**kwargs):
        self.calibration,self.smoke=calibration,smoke;self.evaluations=[]
        data=StoriesData(cfg.data.path,require_pinned=not smoke)
        super().__init__(cfg,data=data,sampler_factory=lambda d,c:StoryStream(d,c,calibration=calibration),**kwargs)
    def validate(self,max_batches=None):
        value=score(self.model,self.data,'cal_val' if self.calibration else 'val',self.config.data.seq_len,
                    self.config.data.eval_batch_size,self.device)
        cost=self._evaluation_cost()
        if cost is not None:self.counters['validation_flops_estimate']+=cost*value['windows']
        self._update_experiment_wall();self._update_peak_memory()
        point={'step':self.counters['step'],'training_bytes':self.counters['tokens_processed'],
            'training_wall_seconds':self.counters['train_wall_seconds'],'experiment_wall_seconds':self.counters['experiment_wall_seconds'],**value}
        self.evaluations.append(point);write_json(self.output_dir/'story-progress.json',point)
        return {'validation_loss':value['nll'],'bits_per_byte':value['bits_per_byte'],'validation_bytes':value['bytes']}


def calibrate(variant,data_root,campaign_root,*,smoke,device,deadline):
    folder=Path(campaign_root)/'calibration'/variant;selection=folder/'selection.json'
    data=StoriesData(data_root,require_pinned=not smoke);source=source_provenance()['source_sha256'];runtime=environment()
    if selection.exists():
        value=json.loads(selection.read_text(encoding='utf-8'))
        if (value['protocol_id']!=PROTOCOL_ID or value['dataset_sha256']!=data.fingerprint or value['source_sha256']!=source or
            value['lr'] not in RATES or any(value['environment'].get(k)!=runtime.get(k) for k in ('python','packages','cuda_version','cudnn_version'))):
            raise ValueError('calibration selection provenance mismatch')
        if (len(value['trials'])!=len(RATES) or {v['lr'] for v in value['trials']}!=set(RATES) or
            any(not math.isfinite(v['nll']) or v['evaluation']['split']!='cal_val' or
                v['evaluation']['bytes']!=data.manifest['splits']['cal_val']['bytes'] or
                v['nll']!=v['evaluation']['nll'] or v['checkpoint_metadata']['dataset_sha256']!=data.fingerprint or
                v['checkpoint_metadata']['provenance']['source_sha256']!=source for v in value['trials']) or
            value['lr']!=min(value['trials'],key=lambda v:(v['nll'],v['lr']))['lr']):
            raise ValueError('calibration selection/trials mismatch')
        return value
    values=[]
    for rate in RATES:
        if deadline and time.monotonic()+30>=deadline:raise TimeoutError('calibration deadline')
        cfg=configuration(variant,data_root,folder/str(rate),331,lr=rate,smoke=smoke,device=device,calibration=True)
        trial_started=time.monotonic()
        with StoryTrainer(cfg,calibration=True,smoke=smoke) as trainer:
            trainer.fit();meta=read_checkpoint_metadata(trainer.last_checkpoint)
            value={'lr':rate,'nll':trainer.evaluations[-1]['nll'],'evaluation':trainer.evaluations[-1],
                'initialization_scope':'seed931; fresh init, no main seed weights','checkpoint_metadata':meta,'counters':trainer.counters}
            value['wall_seconds']=time.monotonic()-trial_started
            values.append(value);write_json(folder/'progress.json',{'completed':len(values),'last':{'lr':rate,'nll':value['nll']}})
        # Only this disposable calibration checkpoint; retain scientific metadata.
        checkpoint=Path(trainer.last_checkpoint);root=folder.resolve()
        if checkpoint.is_symlink() or not checkpoint.resolve().is_relative_to(root):raise ValueError('calibration checkpoint outside own folder')
        meta=read_checkpoint_metadata(checkpoint)
        if set(p.name for p in checkpoint.iterdir())!={'metadata.json',*meta['files']}:raise ValueError('unknown calibration payload')
        for name in meta['files']:
            path=checkpoint/name
            if path.is_symlink():raise ValueError('linked calibration payload')
            path.unlink()
        value['calibration_state_retired']=True
    chosen=min(values,key=lambda v:(v['nll'],v['lr']))
    value={'protocol_id':PROTOCOL_ID,'dataset_sha256':data.fingerprint,'source_sha256':source,'environment':runtime,
        'lr':chosen['lr'],'trials':values,'selection_split':'train-derived cal_val, never official development/test'}
    write_json(selection,value);return value


@torch.no_grad()
def generations(trainer,*,smoke=False):
    model=trainer.model;was=model.training;model.eval();result=[];device=trainer.device
    try:
        for index,prompt in enumerate(PROMPTS[:1] if smoke else PROMPTS):
            for mode in ('greedy','sample'):
                generator=torch.Generator(device=device).manual_seed(13931+index)
                ids=torch.tensor([[256,*prompt.encode('utf-8')]],device=device);made=[];started=time.monotonic()
                for _ in range(8 if smoke else 256):
                    logits=model(ids[:,-trainer.config.data.seq_len:]).logits[0,-1,:256].float()
                    if mode=='greedy':token=int(logits.argmax())
                    else:
                        values,indices=torch.topk(logits,40);chosen=torch.multinomial((values/.8).softmax(-1),1,generator=generator)
                        token=int(indices[chosen])
                    made.append(token);ids=torch.cat((ids,torch.tensor([[token]],device=device)),1)
                raw=bytes(made);text=raw.decode('utf-8',errors='replace')
                result.append({'prompt':prompt,'mode':mode,'seed':13931+index,'new_bytes':len(raw),'text':text,
                    'bytes_hex':raw.hex(),'utf8_replacements':text.count('\ufffd'),'wall_seconds':time.monotonic()-started,
                    'scope':'fixed byte count, no EOS output; uncached recompute with common context cap; qualitative only'})
    finally:model.train(was)
    return result


@torch.no_grad()
def forward_profile(trainer,length,*,smoke=False,model=None):
    model=trainer.model if model is None else model;was=model.training;model.eval();device=trainer.device
    ids=torch.randint(0,256,(1,length),generator=torch.Generator().manual_seed(17731+length)).to(device)
    try:
        with attention_context(device.type=='cpu'):
            for _ in range(1 if smoke else 3):out=model(ids)
            state_bytes=state_storage_bytes(out.state);del out;sync(device)
            baseline=torch.cuda.memory_allocated(device) if device.type=='cuda' else None
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            meter=EnergyMeter(device);start=meter.energy_mj();begin=time.monotonic();power=[];samples=[];count=0;watts=meter.power_watts()
            if watts is not None:power.append((begin,watts))
            try:
                while count<(2 if smoke else 30) or (not smoke and (start is not None or watts is not None) and time.monotonic()-begin<3):
                    sync(device);t=time.monotonic();model(ids);sync(device)
                    if count<(2 if smoke else 30):samples.append(time.monotonic()-t)
                    count+=1;watts=meter.power_watts()
                    if watts is not None:power.append((time.monotonic(),watts))
                energy=energy_result(start,meter.energy_mj(),power,count*length,time.monotonic()-begin)
            finally:meter.close()
        energy['gpu_joules_per_input_byte']=energy.pop('gpu_joules_per_training_target');energy['measured_input_bytes']=energy.pop('measured_training_targets')
        energy['scope']='GPU board repeated full forward; no host/compile; input bytes, not scored targets'
        peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None
        return {'length':length,'median_forward_seconds':statistics.median(samples),'samples_seconds':samples,
            'returned_state_bytes':state_bytes,'peak_vram_bytes':peak,'baseline_vram_bytes':baseline,
            'incremental_peak_bytes':peak-baseline if peak is not None else None,'energy':energy,'host_memory':host_memory(),
            'scope':'eager full forward on fixed random bytes; batch1; not generation/KV or language evaluation'}
    finally:model.train(was)


def optimization_probe(trainer,*,smoke=False):
    if not hasattr(trainer.model,'operators'):return None
    reference=copy.deepcopy(trainer.model).eval();reference.research.associative_execution='reference';reference.architecture_version=1
    length=64 if smoke else 8192;ids=torch.randint(0,256,(1,length),generator=torch.Generator().manual_seed(18831)).to(trainer.device)
    was=trainer.model.training;trainer.model.eval()
    with torch.no_grad(),attention_context(trainer.device.type=='cpu'):
        expected=reference(ids).logits;actual=trainer.model(ids).logits
        torch.testing.assert_close(actual,expected,atol=2e-4,rtol=2e-3)
        error=float((actual-expected).abs().max())
    trainer.model.train(was)
    result={'scope':'explicit weights-only reference clone, no optimizer transfer or resume',
        'max_abs_logit_error':error,'reference':forward_profile(trainer,length,smoke=smoke,model=reference),
        'blocked':forward_profile(trainer,length,smoke=smoke)}
    del reference;return result


def run(config,variant,campaign_root,*,smoke=False,deadline=None,permit=None):
    original=load_config(config);start=time.monotonic()
    try:selection=calibrate(variant,original.data.path,campaign_root,smoke=smoke,device=original.train.device,deadline=deadline)
    except TimeoutError:
        output=Path(original.train.output_dir);output.mkdir(parents=True,exist_ok=False)
        value={'protocol_id':PROTOCOL_ID,'dataset_sha256':StoriesData(original.data.path,require_pinned=not smoke).fingerprint,
            'source_sha256':source_provenance()['source_sha256'],'config_resolved':original.to_dict(),'status':'deferred_budget'}
        write_json(output/'worker.json',value);return value
    cfg=ExperimentConfig.from_dict(original.to_dict());cfg.optimizer.lr=selection['lr'];cfg=ExperimentConfig.from_dict(cfg.to_dict())
    resolved=Path(campaign_root)/'configs'/(Path(cfg.train.output_dir).name+'-resolved.yaml')
    materialize(cfg,resolved)
    with StoryTrainer(cfg,smoke=smoke) as trainer:
        initial=initial_digest(trainer.model)
        pilot=training_benchmark(trainer,samples=2 if smoke else 12,warmup=1 if smoke else 3)
        value=score(trainer.model,trainer.data,'val',cfg.data.seq_len,cfg.data.eval_batch_size,trainer.device)
        estimate=0 if smoke else 1.35*(cfg.train.max_steps*pilot['median_step_seconds']+5*value['wall_seconds']+180)
        result={'protocol_id':PROTOCOL_ID,'dataset_sha256':trainer.data.fingerprint,'source_sha256':trainer.provenance['source_sha256'],
            'environment':trainer.environment_info,'parameters':trainer.model_parameters,'config_resolved':cfg.to_dict(),
            'resolved_config_file':str(resolved),
            'learning_rate':selection['lr'],'calibration':selection,'initial_parameters_sha256':initial,'initial_score':value,'pilot':pilot,
            'forecast_remaining_seconds':estimate,'status':'running','flops_estimate':trainer.flops}
        output=Path(cfg.train.output_dir);write_json(output/'forecast.json',result);write_json(output/'worker.json',result)
        if deadline and time.monotonic()+estimate>=deadline:
            result['status']='deferred_budget';write_json(output/'worker.json',result);return result
        if trainer.device.type=='cuda':torch.cuda.reset_peak_memory_stats(trainer.device)
        trainer.fit();meta=read_checkpoint_metadata(trainer.last_checkpoint)
        if trainer.counters['step']!=cfg.train.max_steps:raise RuntimeError('story worker missed fixed horizon')
        retain_final(output,trainer.last_checkpoint)
        result.update(status='trained',step=trainer.counters['step'],counters=trainer.counters,evaluations=trainer.evaluations,
            stop_reason=trainer.stop_reason,checkpoint=str(trainer.last_checkpoint),checkpoint_model_sha256=meta['files']['model.safetensors'])
        write_json(output/'worker.json',result)
        result['generation']=generations(trainer,smoke=smoke)
        write_json(output/'profile-ready.json',{'status':'ready'})
        while permit and not Path(permit).exists():
            if deadline and time.monotonic()>=deadline:raise TimeoutError('story profile isolation deadline')
            time.sleep(.25)
        result['training_profile']=training_benchmark(trainer,samples=2 if smoke else 30,warmup=1 if smoke else 3)
        result['profiles']=[forward_profile(trainer,n,smoke=smoke) for n in ([32,64] if smoke else [256,1024,8192])]
        result['optimization_probe']=optimization_probe(trainer,smoke=smoke)
        result.update(status='complete',worker_wall_seconds=time.monotonic()-start,final_parameters_sha256=initial_digest(trainer.model))
        write_json(output/'worker.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--variant',required=True);p.add_argument('--campaign-root',required=True)
    p.add_argument('--mode',choices=['train'],default='train');p.add_argument('--engineering-smoke',action='store_true');p.add_argument('--work-deadline',type=float);p.add_argument('--profile-permit')
    a=p.parse_args();run(a.config,a.variant,a.campaign_root,smoke=a.engineering_smoke,deadline=a.work_deadline,permit=a.profile_permit)


if __name__=='__main__':main()
