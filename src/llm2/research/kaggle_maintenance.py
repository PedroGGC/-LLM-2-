"""Standard-library maintenance: inventories data, removes only clean code caches."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import zipfile

REMOTE='https://github.com/PedroGGC/-LLM-2-.git'
CACHE_NAME=re.compile(r'llm2-cycle2-(?:git|code)-[A-Za-z0-9_-]+')
PROTECTED_SUFFIXES={'.safetensors','.pt','.pth','.ckpt','.bin','.parquet'}
SCIENCE_IDENTITY_SHA256='0ce75e923da1af584f3d1fedf2ddf2c4e1af06bd709e248f9666b465f5b5028e'
FLRM_103_SHA256='54cbce20834855ad6b2de0c4bb3ebb13ffbaf8766ac7b56d2ae0b9269835da7c'


def size(path):
    path=Path(path)
    if path.is_symlink():return 0
    if path.is_file():return path.stat().st_size
    total=0
    for folder,dirs,files in os.walk(path,followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
        for name in files:
            try:total+=(Path(folder)/name).lstat().st_size
            except OSError:pass
    return total


def active_workers():
    found=[]
    for p in Path('/proc').glob('[0-9]*'):
        if int(p.name)==os.getpid():continue
        try:
            args=[s.decode('utf-8','replace') for s in (p/'cmdline').read_bytes().split(b'\0') if s]
        except OSError:continue
        if any(s=='llm2' or s.startswith('llm2.') or 'scripts/run_cycle' in s
               or '/llm2-cycle2-git-' in s or '/llm2-cycle2-code-' in s for s in args):
            found.append({'pid':int(p.name),'program':Path(args[0]).name if args else 'unknown'})
    return found


def git(path,*args):
    return subprocess.check_output(['git','-C',str(path),*args],stderr=subprocess.STDOUT,timeout=15).decode('utf-8').strip()


def code_cache_targets(parent,base):
    """Every destructive target is resolved/checked; data and dirty repos veto the cache."""
    parent=Path(parent);base=Path(base).resolve()
    if parent.is_symlink() or parent.resolve().parent!=base or not CACHE_NAME.fullmatch(parent.name):
        raise ValueError('not an owned code-cache parent')
    roots=[parent/name for name in ('scientific-worker','source') if (parent/name).exists()]
    if not roots:raise ValueError('no code checkouts')
    for root in roots:
        if root.is_symlink() or root.resolve().parent!=parent.resolve():raise ValueError('external checkout path')
        if git(root,'remote','get-url','origin')!=REMOTE:raise ValueError('unexpected remote')
        git(root,'rev-parse','--verify','HEAD')
        common=Path(git(root,'rev-parse','--git-common-dir'))
        common=(root/common).resolve() if not common.is_absolute() else common.resolve()
        if not common.is_relative_to(parent.resolve()):raise ValueError('Git common directory outside cache')
        if git(root,'status','--porcelain','--untracked-files=all'):raise ValueError('modified/untracked files')
        ignored=git(root,'ls-files','--others','--ignored','--exclude-standard','-z').split('\0')
        for name in filter(None,ignored):
            if not any(part in ('__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','.ipynb_checkpoints')
                       or part.endswith('.egg-info') for part in Path(name).parts):
                raise ValueError('ignored files beyond reproducible Python caches')
        for path in root.rglob('*'):
            if path.is_symlink():raise ValueError('symlink in checkout')
            if path.is_file() and path.suffix.lower() in PROTECTED_SUFFIXES:
                raise ValueError('data/checkpoint payload in checkout')
    return roots


def remove_cache(target):
    target=Path(target).resolve()
    def readonly(function,path,exception):
        candidate=Path(path)
        if (os.name!='nt' or not isinstance(exception[1],PermissionError) or candidate.is_symlink()
                or not candidate.resolve().is_relative_to(target)):
            raise exception[1]
        candidate.chmod(candidate.stat().st_mode|stat.S_IWRITE)
        function(path)
    shutil.rmtree(target,onerror=readonly)


def next_number(base,input_root=None):
    pattern=re.compile(r'llm2-campaign-(\d+)(?:-.*)?$')
    numbers=[21]
    paths=list(Path(base).glob('llm2-campaign-*'))
    if input_root is not None:paths+=list(Path(input_root).rglob('llm2-campaign-*'))
    for path in paths:
        match=pattern.fullmatch(path.name)
        if match:numbers.append(int(match[1]))
    registry=Path(base)/'.llm2-numbering/llm2-campaign'
    if registry.exists():numbers += [int(p.name) for p in registry.iterdir() if p.name.isdecimal()]
    return max(numbers)+1


def campaign_state(base,number=21):
    directory=Path(base)/f'llm2-campaign-{number:04d}'
    meta=directory/'campaign.json'
    if not meta.exists():return [],[]
    try:
        if meta.stat().st_size>10*1024**2:raise ValueError('oversized metadata')
        value=json.loads(meta.read_text(encoding='utf-8'))
    except (OSError,ValueError) as error:
        return [],[f'Não consegui conferir {meta}: {error}']
    if not isinstance(value,dict) or not isinstance(value.get('runs'),list):
        return [],['Manifesto de campanha inválido; preservar e verificar antes de continuar.']
    exported={};warnings=[]
    for archive_path in Path(base).glob(f'llm2-campaign-{number:04d}*-results.zip'):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                name='report/report.json'
                if archive.getinfo(name).file_size>10*1024**2:continue
                report=json.loads(archive.read(name))
                identity_hash=hashlib.sha256(json.dumps(report.get('identity'),sort_keys=True).encode()).hexdigest()
                if identity_hash!=SCIENCE_IDENTITY_SHA256:continue
                if report.get('engineering_smoke') or report.get('test_evaluated'):continue
                for row in report.get('rows',[]):
                    exported[(row['arm'],row['seed'])]=row['checkpoint_model_sha256']
        except (OSError,ValueError,KeyError,zipfile.BadZipFile) as error:
            warnings.append(f'Histórico ilegível em {archive_path}: {error}')
    rows=[]
    for job in value.get('runs',[]):
        if not isinstance(job,dict):warnings.append('Registro de treino inválido');continue
        identity=job.get('id','')
        if not re.fullmatch(r'[a-z0-9_-]+',identity):
            warnings.append('Identificador de treino inválido no manifesto');continue
        run=directory/'runs'/identity
        states=sorted((run/'checkpoints').glob('step-*'))
        complete=job.get('status')=='complete'
        in_export=exported.get((job.get('arm'),job.get('seed')))==job.get('checkpoint_model_sha256') and complete
        row={'id':identity,'status':job.get('status'),'step':job.get('step'),
             'last_checkpoint':states[-1].name if states else None,'complete_in_results_zip':in_export}
        rows.append(row)
        if states and not complete:warnings.append(f'{identity}: checkpoint parcial/aguardando avaliação; preservar e recuperar antes de reiniciar.')
        elif complete and not in_export:warnings.append(f'{identity}: concluído sem ZIP de resultados correspondente; exportar antes de continuar.')
    return rows,warnings


def known_flrm_103(base,input_root):
    paths=list(Path(base).glob('llm2-campaign-*-results.zip'))
    if input_root is not None:paths+=list(Path(input_root).rglob('llm2-campaign-*-results.zip'))
    for path in paths:
        try:
            with zipfile.ZipFile(path) as archive:
                name='history/report.json' if 'history/report.json' in archive.namelist() else 'report/report.json'
                if archive.getinfo(name).file_size>10*1024**2:continue
                report=json.loads(archive.read(name))
                if not isinstance(report,dict):continue
                identity=hashlib.sha256(json.dumps(report.get('identity'),sort_keys=True).encode()).hexdigest()
                if identity!=SCIENCE_IDENTITY_SHA256 or report.get('engineering_smoke') or report.get('test_evaluated'):continue
                rows=report.get('rows',[])
                if not isinstance(rows,list):continue
                if any(isinstance(r,dict) and r.get('arm')=='anchor_flrm' and r.get('seed')==103
                       and r.get('checkpoint_model_sha256')==FLRM_103_SHA256 for r in rows):return True
        except (OSError,ValueError,KeyError,zipfile.BadZipFile):continue
    return False


def clean_working(base='/kaggle/working',*,apply=False,input_root='/kaggle/input'):
    base=Path(base).resolve()
    if not base.is_dir():raise ValueError('Execute no Kaggle; diretório de trabalho ausente')
    before=shutil.disk_usage(base);workers=active_workers()
    states,warnings=campaign_state(base)
    have_flrm=known_flrm_103(base,input_root)
    if not have_flrm:warnings.append('Anexe/preserve o ZIP after-01 de resultados da 0021 antes de continuar; o FLRM 103 nao deve ser repetido.')
    inventory=[]
    for p in base.iterdir():
        count=size(p);inventory.append({'name':p.name,'bytes':count,'gib':round(count/1024**3,3)})
    inventory.sort(key=lambda r:r['gib'],reverse=True)
    print('Maior consumo em /working:')
    for row in inventory[:25]:print(f"  {row['gib']:7.3f} GiB  {row['name']}")
    removed=[];preserved=[]
    if workers:print('Há workers ativos; limpeza desativada:',workers)
    else:
        for parent in base.iterdir():
            if not CACHE_NAME.fullmatch(parent.name):continue
            try:targets=code_cache_targets(parent,base)
            except (ValueError,OSError,subprocess.SubprocessError) as error:
                preserved.append({'name':parent.name,'reason':str(error)});continue
            for path in targets:
                target=path.resolve()
                if path.is_symlink() or target.parent!=parent.resolve() or not target.is_relative_to(base):
                    raise ValueError('target changed before cleanup')
                count=size(path)
                if apply:remove_cache(target)
                removed.append({'path':str(target),'gib':round(count/1024**3,3),'deleted':apply})
                print(('Removido código reproduzível: ' if apply else 'Pode remover código reproduzível: ')+str(path))
    after=shutil.disk_usage(base)
    result={'working_before_gib':round(sum(r['bytes'] for r in inventory)/1024**3,3),
            'working_after_gib':round(size(base)/1024**3,3),
            'filesystem_free_gib':round(after.free/1024**3,3),
            'filesystem_free_delta_gib':round((after.free-before.free)/1024**3,3),'active_workers':workers,
            'campaign_0021':states,'warnings':warnings,'code_caches':removed,'protected_caches':preserved,
            'verified_flrm_103_result_available':have_flrm,
            'next_run_number':next_number(base,input_root),'safe_to_start_new_batch':not workers and not warnings}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result
