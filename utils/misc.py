import csv
import json
import logging
import os
import random
from pathlib import Path
import numpy as np
import torch
import yaml


def config(path):
    with open(path,encoding='utf8') as fp:
        c=yaml.safe_load(fp)
    s=c['semantic']
    if any(int(s[t])<1 for t in ('n','d','k','m')) or s['k']<2:
        raise ValueError('Require N,D>=1 and K>=2')
    if s['power']<=0:
        raise ValueError('Power must be > 0')
    return c


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def logger(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    log=logging.getLogger(str(out.resolve()));log.handlers.clear();log.propagate=False;log.setLevel(logging.INFO)
    fmt=logging.Formatter('%(asctime)s | %(message)s','%Y-%m-%d %H:%M:%S')
    for handler in (logging.StreamHandler(),logging.FileHandler(out/'train.log',encoding='utf8')):
        handler.setFormatter(fmt);log.addHandler(handler)
    return log


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    torch.save(obj,tmp);os.replace(tmp,path)


def write_csv(path,rows,append=False):
    rows=list(rows)
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    exists=append and path.exists() and path.stat().st_size>0
    with path.open('a' if append else 'w',newline='',encoding='utf8') as fp:
        writer=csv.DictWriter(fp,list(rows[0]))
        if not exists:writer.writeheader()
        writer.writerows(rows)


def make_generators(device,seed):
    return tuple(torch.Generator(device=device.type).manual_seed(int(seed+i*10000019))
                 for i in range(3))


def channel_seed(seed,kind,snr,repeat):
    return int(seed+1000003*{'awgn':1,'rician':2,'rayleigh':3}[kind]
               +int(round((float(snr)+100)*100))*127+repeat*27011)


def state_for_save(model,opt,sched,scaler,cfg,classes,backbone,stage,epoch,best,best_epoch,
                   phase='hard',vq_initialized=True):
    return dict(model=model.state_dict(),optimizer=opt.state_dict(),
                scheduler=sched.state_dict(),scaler=scaler.state_dict() if scaler else None,
                config=cfg,classes=classes,backbone=backbone,stage=stage,epoch=epoch,
                best=best,best_epoch=best_epoch,rng=rng_state(),
                phase=phase,vq_initialized=bool(vq_initialized),format_version=5)


def snapshot_config(out, config_obj):
    (Path(out)/'settings.json').write_text(json.dumps(config_obj,indent=2,ensure_ascii=False),encoding='utf8')
