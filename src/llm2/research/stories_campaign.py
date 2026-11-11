"""Bounded first language comparison; private caches and final-only publication."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import time
import traceback
STARTED=time.monotonic()
import torch
from ..checkpoint import read_checkpoint_metadata,resume_contract
from ..config import ExperimentConfig
from ..data import write_json,sha256_file
from ..runtime import environment,save_source_snapshot
from .stories_data import prepare,StoriesData,write_dataset,PINNED_FINGERPRINT
from .stories_protocol import VARIANTS,SEEDS,RATES,PROTOCOL,PROTOCOL_ID,configuration,jobs
from . import stories_protocol as DEFAULT_RECIPE
from .stories_ngram import evaluate_ngram
from .campaign_storage import verify_pair
from .cycle3_short import heartbeat
from .cycle4 import archive_files,checkpoint_files
from .session import SessionBudget,SessionBudgetExceeded,SessionCampaign
from .state_study import materialize,storage_budget,discard_compiler_cache
ROOT=Path(__file__).resolve().parents[3]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def smoke_data(path):
    splits={}
    for split in ('train','cal_val','val','test'):
        values=[(f'{split}: Once upon a time, a little fox found a red ball. {i}\n').encode()*(i+1) for i in range(2)]
        splits[split]={hashlib.sha256(v).hexdigest():v for v in values}
    return write_dataset(path,splits,spec={'engineering_smoke':True})


def history(campaign,args,data,*,recipe=DEFAULT_RECIPE):
    PROTOCOL_ID,RATES,VARIANTS,configuration=recipe.PROTOCOL_ID,recipe.RATES,recipe.VARIANTS,recipe.configuration
    found={};runtime=environment()
    for path in sorted(campaign.output.parent.glob('llm2-campaign-*/campaign.json')):
        if path.parent==campaign.output or path.is_symlink():continue
        try:old=read(path)
        except (ValueError,OSError):continue
        if old.get('cycle')!=recipe.CYCLE or old.get('protocol_id')!=PROTOCOL_ID or old.get('engineering_smoke')!=args.engineering_smoke:continue
        if old['dataset_sha256']!=data.fingerprint or old['provenance']['source_sha256']!=campaign.manifest['provenance']['source_sha256']:
            raise ValueError('story history dataset/source mismatch')
        if any(old['environment'].get(k)!=runtime.get(k) for k in ('python','packages','cuda_version','cudnn_version')):raise ValueError('story history runtime mismatch')
        for job in old['runs']+old.get('historical_runs',[]):
            if job['status']!='complete':continue
            if job['learning_rate'] not in RATES:raise ValueError('historical rate outside frozen calibration')
            expected=configuration(job['variant'],data.root,'unused',job['seed'],lr=job['learning_rate'],smoke=args.engineering_smoke,device=args.device)
            if resume_contract(expected)!=resume_contract(ExperimentConfig.from_dict(job['config_resolved'])):raise ValueError('story history config mismatch')
            origin=Path(job.get('historical_campaign',path.parent));cp=Path(job['checkpoint'])
            if origin.is_symlink() or cp.is_symlink() or not origin.resolve().is_relative_to(campaign.output.parent.resolve()) or not cp.resolve().is_relative_to(origin.resolve()):raise ValueError('story checkpoint outside working')
            meta=read_checkpoint_metadata(cp)
            if (job['step']!=expected.train.max_steps or meta['config']!=job['config_resolved'] or meta['dataset_sha256']!=data.fingerprint or
                meta['files']['model.safetensors']!=job['checkpoint_model_sha256'] or meta['provenance']['source_sha256']!=campaign.manifest['provenance']['source_sha256']):
                raise ValueError('story history checkpoint mismatch')
            if job['id'] in found and found[job['id']]['checkpoint_model_sha256']!=job['checkpoint_model_sha256']:raise ValueError('conflicting story history')
            found[job['id']]={**job,'historical_campaign':str(origin),'history_manifest_sha256':sha256_file(origin/'campaign.json')}
    for variant in VARIANTS:
        selections=[j['calibration'] for j in found.values() if j['variant']==variant]
        if selections:
            if len({s['lr'] for s in selections})!=1:raise ValueError('conflicting calibrated rate')
            folder=campaign.output/'calibration'/variant;folder.mkdir(parents=True);write_json(folder/'selection.json',selections[0])
    return list(found.values())


def run_pair(campaign,args,pair,*,recipe=DEFAULT_RECIPE):
    PROTOCOL_ID=recipe.PROTOCOL_ID
    def run(job):
        job['status']='running';campaign.save();cfg=ExperimentConfig.from_dict(job['config_resolved'])
        path=materialize(cfg,campaign.output/'configs'/(job['id']+'-proposal.yaml'))
        command=['-m',recipe.WORKER_MODULE,'--config',path,'--variant',job['variant'],'--mode','train',
            '--campaign-root',str(campaign.output),'--work-deadline',str(time.monotonic()+campaign.budget.work_remaining),
            '--profile-permit',str(Path(job['directory'])/'profile-permit.json')]
        if args.engineering_smoke:command.append('--engineering-smoke')
        started=time.monotonic();campaign.command(command,job['id']+'.log',gpu=None if args.engineering_smoke else job['gpu_slot'],job=job)
        value=read(Path(job['directory'])/'worker.json')
        if value['protocol_id']!=PROTOCOL_ID or value['dataset_sha256']!=campaign.manifest['dataset_sha256'] or value['source_sha256']!=campaign.manifest['provenance']['source_sha256']:
            raise ValueError('story worker identity mismatch')
        if value['status']=='complete' and value['step']!=job['max_steps']:raise ValueError('story worker horizon mismatch')
        with campaign.lock:job.update(value,command_wall_seconds=time.monotonic()-started);campaign.save()
    active=None;reported={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={j['id']:pool.submit(run,j) for j in pair}
        try:
            while not all(f.done() for f in futures.values()):
                for f in futures.values():
                    if f.done() and f.exception() is not None:f.result()
                for j in pair:
                    paths=[('calibration',campaign.output/'calibration'/j['variant']/'progress.json'),
                           ('pilot',Path(j['directory'])/'forecast.json'),('score',Path(j['directory'])/'story-progress.json')]
                    for kind,path in paths:
                        if not path.exists():continue
                        try:value=read(path)
                        except (ValueError,OSError):continue
                        index=value.get('step',value.get('completed',1));key=(j['id'],kind)
                        if reported.get(key)==index:continue
                        reported[key]=index
                        if kind=='calibration':print(f"CALIBRAÇÃO {j['variant']}: {value['last']}",flush=True)
                        elif kind=='pilot':print(f"PILOTO {j['id']}: LR{value['learning_rate']}, {1000*value['pilot']['median_step_seconds']:.2f}ms/passo; previsão{value['forecast_remaining_seconds']/60:.1f}min",flush=True)
                        else:print(f"LINGUAGEM {j['id']} passo{value['step']}: NLL{value['nll']:.4f} | {value['bits_per_byte']:.4f}bits/byte",flush=True)
                if active and futures[active].done():active=None
                if active is None and all(futures[j['id']].done() or (Path(j['directory'])/'profile-ready.json').exists() for j in pair):
                    job=next((j for j in pair if not futures[j['id']].done()),None)
                    if job:
                        active=job['id'];write_json(Path(job['directory'])/'profile-permit.json',{'scope':'both lanes finished training/generation; one profile at a time'})
                        print('PERFIL ISOLADO: '+active,flush=True)
                time.sleep(.25)
            for f in futures.values():f.result()
        except BaseException:campaign.abort();raise


def report(campaign,*,recipe=DEFAULT_RECIPE):
    VARIANTS,SEEDS,PROTOCOL_ID=recipe.VARIANTS,recipe.SEEDS,recipe.PROTOCOL_ID
    evidence=campaign.manifest.get('historical_runs',[])+campaign.manifest['runs'];rows=[];curves=[]
    for j in evidence:
        for e in j.get('evaluations',[]):curves.append({'variant':j['variant'],'seed':j['seed'],**e})
        if not j.get('evaluations'):continue
        e=j['evaluations'][-1];p=next((p for p in j.get('profiles',[]) if p['length']==j['config_resolved']['data']['seq_len']),{})
        rows.append({'variant':j['variant'],'seed':j['seed'],'status':j['status'],'parameters':j.get('parameters'),
            'lr':j.get('learning_rate'),'nll':e['nll'],'bits_per_byte':e['bits_per_byte'],'validation_bytes':e['bytes'],
            'training_bytes':j['counters']['tokens_processed'],'train_wall_seconds':j['counters']['train_wall_seconds'],
            'train_step_ms':1000*j['training_profile']['median_step_seconds'] if j.get('training_profile') else None,
            'forward_ms':1000*p['median_forward_seconds'] if p else None,'training_vram_bytes':j['counters']['peak_vram_allocated_bytes'],
            'gpu_slot':j['gpu_slot'],'complete_horizon':j['status']=='complete' and j['step']==j['config_resolved']['train']['max_steps']})
    comparisons=[]
    for row in rows:
        if row['variant'].startswith('transformer'):continue
        for name in recipe.REFERENCES:
            base=next((r for r in rows if r['variant']==name and r['seed']==row['seed']),None)
            if base is None:continue
            difference=row['nll']-base['nll']
            time_factor=base['train_step_ms']/row['train_step_ms'] if base['train_step_ms'] and row['train_step_ms'] else None
            comparisons.append({'variant':row['variant'],'seed':row['seed'],'reference':name,'nll_difference':difference,
                'complete_horizon':row['complete_horizon'] and base['complete_horizon'],
                'parameter_factor':base['parameters']/row['parameters'],'train_step_factor':time_factor,
                'quality_candidate':row['complete_horizon'] and base['complete_horizon'] and difference<=-.005,
                'efficiency_candidate':bool(row['complete_horizon'] and base['complete_horizon'] and difference<=.002 and time_factor and time_factor>=1/.9),
                'scope':'exploratory per-seed same-input result; not matched compute or independent confirmation'})
    folder=campaign.output/'report';folder.mkdir(exist_ok=True)
    milestones=[]
    for variant in VARIANTS:
        if variant.startswith('transformer'):continue
        for reference in recipe.REFERENCES:
            pairs=[r for r in comparisons if r['variant']==variant and r['reference']==reference]
            for factor in recipe.PROTOCOL.get('cost_targets',[]):
                passed={p['seed'] for p in pairs}==set(SEEDS) and all(p['complete_horizon'] and p['nll_difference']<=.002 and p['train_step_factor'] is not None and p['train_step_factor']>=factor for p in pairs)
                milestones.append({'variant':variant,'reference':reference,'target_speed_factor':factor,'passed_both_seeds':passed,
                    'scope':'isolated update speed at final NLL within+0.002; same targets, exploratory, not confirmed time-to-quality'})
    value={'protocol_id':PROTOCOL_ID,'status':campaign.manifest['status'],'dataset_sha256':campaign.manifest.get('dataset_sha256'),
        'rows':rows,'comparisons':comparisons,'counting_baseline':campaign.manifest.get('counting_baseline'),
        'resource_milestones':milestones,
        'test_evaluated':False,'limitations':['reduced deterministic prefix subset, byte-level English generated stories',
            'two seeds, fixed finite LR calibration budget, no general LLM superiority','same sampled windows, not matched compute',
            'generation qualitative with fixed prompts, no cherrypicked examples','forward benchmark not cached generation','GPU board energy excludes host/compile']}
    write_json(folder/'report.json',value)
    for label,data in (('scores',rows),('curves',curves)):
        if data:
            with (folder/(label+'.csv')).open('w',encoding='utf-8',newline='') as stream:
                w=csv.DictWriter(stream,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
    keys=('variant','seed','parameters','lr','nll','bits_per_byte','train_step_ms')
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in keys)+'</tr>' for r in rows)
    page='<meta charset="utf-8"><title>TinyStories language screen</title><style>body{font:16px sans-serif;margin:30px}td,th{padding:8px;border-bottom:1px solid #ccc}pre{white-space:pre-wrap;background:#f3f3f3;padding:16px}</style>'
    page+='<h1>Linguagem: perda e textos</h1><p>Menor NLL/bits por byte é melhor. Histórias geradas pelo corpus; avaliação de desenvolvimento, teste reservado intacto.</p>'
    count=campaign.manifest.get('counting_baseline')
    if count:page+=f"<p>Referência de contagens de três bytes: {count['nll']:.4f}NLL / {count['bits_per_byte']:.4f}bits porbyte. Estado de contagens: {count['count_state_bytes']}bytes; não é uma rede com parâmetros equivalentes.</p>"
    page+='<table><tr>'
    page+=''.join('<th>'+k+'</th>' for k in keys)+'</tr>'+table+'</table>'
    if milestones:
        passed=[m for m in milestones if m['passed_both_seeds']]
        page+='<h2>Metas de velocidade com qualidade equivalente</h2><p>Exigem os dois treinos completos e NLL até 0,002 pior em cada seed. Tempo isolado por atualização; confirmação independente ainda necessária.</p>'
        if passed:
            page+='<ul>'+''.join('<li>'+html.escape(f"{m['variant']} / {m['reference']}: atingiu {m['target_speed_factor']}x")+'</li>' for m in passed)+'</ul>'
        else:page+='<p>Nenhuma meta qualificada com os resultados completos disponíveis.</p>'
    for j in evidence:
        for sample in j.get('generation',[]):
            page+='<h3>'+html.escape(j['id']+' / '+sample['mode'])+'</h3><pre>'+html.escape(sample['prompt']+sample['text'])+'</pre>'
    (folder/'report.html').write_text(page,encoding='utf-8',newline='\n')
    if curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        colors={v:plt.get_cmap('tab10')(i) for i,v in enumerate(VARIANTS)}
        for variant in VARIANTS:
            for index,seed in enumerate(SEEDS):
                points=[p for p in curves if p['variant']==variant and p['seed']==seed]
                if not points:continue
                for ax,key in zip(axes,('training_bytes','training_wall_seconds')):
                    ax.plot([p[key] for p in points],[p['bits_per_byte'] for p in points],
                        linestyle='-' if index==0 else '--',color=colors[variant],label=variant if index==0 else None)
        for ax in axes:
            ax.set_ylabel('Bits por byte (menor é melhor)');ax.grid(alpha=.18)
        axes[0].set_xlabel('Bytes supervisionados com repetição');axes[1].set_xlabel('Segundos de treino, excluindo calibração/piloto')
        axes[0].legend(fontsize=8);fig.savefig(folder/'quality_cost.png',dpi=150);plt.close(fig)


def publish(campaign,*,recipe=DEFAULT_RECIPE):
    campaign.save();report(campaign,recipe=recipe)
    files=[]
    for p in campaign.output.rglob('*'):
        if not p.is_file() or 'compiler-cache' in p.parts or p.suffix not in ('.json','.jsonl','.yaml','.log','.txt','.csv','.html','.zip','.png'):continue
        if p.name in ('source.zip','tokenizer.json') and p.parent not in (campaign.output,campaign.output/'data'):continue
        if p.name=='dataset.manifest.json':continue
        files.append(p)
    result=archive_files(campaign.output.parent/(campaign.output.name+'-results.zip'),files,base=campaign.output)
    weights=archive_files(campaign.output.parent/(campaign.output.name+'-checkpoints.zip'),checkpoint_files(campaign,campaign.manifest['runs']),base=campaign.output)
    verify_pair(result,weights);return result,weights


def execute(args,*,recipe=DEFAULT_RECIPE):
    PROTOCOL,PROTOCOL_ID,VARIANTS,jobs=recipe.PROTOCOL,recipe.PROTOCOL_ID,recipe.VARIANTS,recipe.jobs
    budget=SessionBudget(args.session_seconds,300,elapsed=args.bootstrap_elapsed_seconds+time.monotonic()-args.started)
    campaign=SessionCampaign(ROOT,args.output,device=args.device,gpus=args.gpus,budget=budget)
    campaign.manifest.update(cycle=recipe.CYCLE,protocol=PROTOCOL,protocol_id=PROTOCOL_ID,engineering_smoke=args.engineering_smoke,
        environment=environment(),historical_runs=[],dataset_sha256=PINNED_FINGERPRINT,
        limitations='first byte-language screen; reduced corpus; test never evaluated; no automatic general superiority claim')
    old={k:os.environ.get(k) for k in ('LLM2_CYCLE4','LLM2_COMPILER_CACHE_ROOT')}
    os.environ.update(LLM2_CYCLE4='1',LLM2_COMPILER_CACHE_ROOT=str(campaign.output/'compiler-cache'));error=None
    try:
        with storage_budget(campaign,smoke=args.engineering_smoke),heartbeat(campaign,'TinyStories e linguagem'):
            if args.bootstrap_metadata:
                metadata=read(args.bootstrap_metadata)
                for name,digest in metadata['source_files_sha256'].items():
                    if sha256_file(ROOT/name)!=digest:raise ValueError('bootstrap source changed: '+name)
                if metadata['source_git_revision']!=campaign.manifest['provenance']['git']['commit']:raise ValueError('bootstrap revision mismatch')
                write_json(campaign.output/'bootstrap.json',metadata)
            save_source_snapshot(campaign.output/'source.zip',campaign.manifest['provenance'])
            if not args.engineering_smoke and shutil.disk_usage(campaign.output).free<4*1024**3+256*1024**2:
                raise OSError('insufficient free space before dataset preparation')
            data=smoke_data(campaign.output/'engineering-data') if args.engineering_smoke else prepare(args.cache,budget=budget)
            campaign.manifest.update(dataset_sha256=data.fingerprint,dataset_path=str(data.root))
            target=campaign.output/'data';target.mkdir();shutil.copyfile(data.root/'manifest.json',target/'manifest.json');shutil.copyfile(data.root/'tokenizer.json',target/'tokenizer.json')
            campaign.manifest['counting_baseline']=evaluate_ngram(data,32 if args.engineering_smoke else 256)
            print('DATASET verificado: '+data.fingerprint,flush=True)
            if not args.engineering_smoke:
                campaign.command(['-m','pytest','-q',*recipe.CPU_TESTS,'-m','not cuda'],'tests-stories-cpu.log')
                campaign.command(['-m','pytest','-q',*recipe.CUDA_TESTS,'-m','cuda'],'tests-stories-cuda.log',gpu=args.gpus[0])
            previous=history(campaign,args,data,recipe=recipe);campaign.manifest['historical_runs']=previous;done={j['id'] for j in previous}
            rows=[j for j in jobs(campaign.output,data.root,args.gpus,seeds=args.seeds,smoke=args.engineering_smoke,device=args.device) if j['id'] not in done]
            campaign.manifest['runs']=rows;campaign.save()
            from .factory import build_backbone
            with torch.device('meta'):
                weight_counts=[sum(p.numel() for p in build_backbone(ExperimentConfig.from_dict(j['config_resolved'])).parameters()) for j in rows]
            required=(0 if args.engineering_smoke else 4*1024**3)+sum(weight_counts)*48
            if shutil.disk_usage(campaign.output).free<required:raise OSError('insufficient space for cache/checkpoints/free reserve')
            campaign.manifest['storage_estimate']={'required_free_bytes':required}
            for seed in args.seeds:
                pending=[j for j in rows if j['seed']==seed]
                while pending:
                    budget.require(0 if args.engineering_smoke else 180)
                    first=pending.pop(0);other=next((j for j in pending if j['gpu_slot']!=first['gpu_slot']),None);pair=[first]
                    if other:pending.remove(other);pair.append(other)
                    run_pair(campaign,args,pair,recipe=recipe)
            complete=[j for j in previous+rows if j['status']=='complete']
            for seed in args.seeds:
                counts={j['counters']['tokens_processed'] for j in complete if j['seed']==seed}
                if len(counts)>1:raise ValueError('unequal sampled language targets between models')
            campaign.manifest['status']='complete' if len(complete)==len(VARIANTS)*len(args.seeds) else 'partial_planned'
    except SessionBudgetExceeded as exc:campaign.manifest.update(status='partial_budget',error=str(exc))
    except Exception as exc:error=exc;campaign.manifest.update(status='failed',error=str(exc),traceback=traceback.format_exc())
    finally:
        campaign.abort()
        for j in campaign.manifest['runs']:
            path=Path(j['directory'])/'worker.json'
            if path.exists():
                try:
                    value=read(path)
                    if value.get('source_sha256')==campaign.manifest['provenance']['source_sha256']:j.update(value)
                except (ValueError,OSError):pass
        try:discard_compiler_cache(campaign)
        except Exception as exc:campaign.manifest['cache_cleanup_error']=str(exc)
        completed={j['id'] for j in campaign.manifest.get('historical_runs',[])+campaign.manifest['runs'] if j['status']=='complete'}
        campaign.manifest['pending_ids']=[f'{v}-seed-{s}' for s in args.seeds for v in VARIANTS if f'{v}-seed-{s}' not in completed]
        campaign.manifest['session_budget']=budget.snapshot()
        try:result,weights=publish(campaign,recipe=recipe)
        finally:
            for k,v in old.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v
        print(f"Status:{campaign.manifest['status']} | {result.name} | {weights.name}",flush=True)
    if error:raise RuntimeError('language campaign failed; evidence exported') from error
    return campaign.manifest


def main(*,recipe=DEFAULT_RECIPE,started=STARTED):
    SEEDS=recipe.SEEDS
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--cache',required=True);p.add_argument('--gpus',nargs=2,default=['0','1'])
    p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS));p.add_argument('--session-seconds',type=float,default=7200)
    p.add_argument('--bootstrap-elapsed-seconds',type=float,default=0);p.add_argument('--bootstrap-metadata')
    p.add_argument('--engineering-smoke',action='store_true');p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    a=p.parse_args();a.started=started
    if not 300<a.session_seconds<=7200 or len(set(a.gpus))!=2 or (a.device=='cpu')!=a.engineering_smoke:raise ValueError('two-hour/two slots required; CPU is engineering only')
    if not a.seeds or len(set(a.seeds))!=len(a.seeds) or not set(a.seeds)<=set(SEEDS):raise ValueError('invalid language seeds')
    if not a.engineering_smoke and (not a.bootstrap_metadata or not torch.cuda.is_available() or any(not s.isdecimal() or int(s)>=torch.cuda.device_count() or 'T4' not in torch.cuda.get_device_name(int(s)) for s in a.gpus)):
        raise ValueError('pinned bootstrap and two T4 GPUs required')
    execute(a,recipe=recipe)


if __name__=='__main__':main()
