import argparse
from pathlib import Path
from statistics import mean, stdev
import torch
from models import System,Channel,MODES,CHANNELS
from utils.data import loaders
from utils.misc import (channel_seed,logger,make_generators,seed_everything,write_csv)


def evaluate(model, val, device, channel=None, kind='awgn', mode='none', snr=0., seed=42,
             amp=False):
    model.eval()
    generators=make_generators(device,seed)
    counters=dict(total=0,top1=0,top5=0,index=0,original_top1=0)
    for images,labels in val:
        images=images.to(device,non_blocking=True)
        labels=labels.to(device,non_blocking=True)
        with torch.no_grad():
            with torch.autocast(device_type=device.type,enabled=amp):
                feat,teacher=model.features(images)
                tx,q=model.codec.encode(feat)
                rx=tx if channel is None else channel(tx,kind,snr,mode,*generators)
                recovered,prob,_=model.codec.decode(rx)
                logits=model.classify(recovered)
        n=labels.numel()
        hit=logits.topk(5,dim=-1).indices.eq(labels[:,None])
        counters['total']+=n
        counters['top1']+=int(hit[:,0].sum())
        counters['top5']+=int(hit.any(dim=-1).sum())
        counters['index']+=int((q['indices']==prob.argmax(dim=-1)).sum())
        counters['original_top1']+=int((teacher.argmax(dim=-1)==labels).sum())
    n=counters['total']
    if n<1:raise ValueError('Empty validation set')
    return dict(top1=100*counters['top1']/n,top5=100*counters['top5']/n,
                index_acc=100*counters['index']/(n*model.codec.encoder.queries.shape[1]),
                original_top1=100*counters['original_top1']/n,samples=n)


def test_grid(model, val, device, cfg, out, log, snrs=None, modes=None, repeats=None, amp=False):
    opts=cfg['test']
    snrs=opts['snrs'] if snrs is None else snrs
    modes=opts['modes'] if modes is None else modes
    repeats=opts['repeats'] if repeats is None else repeats
    if repeats<1:raise ValueError('Test repeats must be >=1')
    if any(m not in MODES for m in modes):raise ValueError('Invalid receiver mode')
    ch=Channel(power=cfg['semantic']['power'],rician_k=opts['rician_k'],
               csi_error_var=opts['csi_error_var'])
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    baseline=evaluate(model,val,device,amp=amp)
    write_csv(out/'clean.csv',[dict(setting='no_channel_clean',**baseline)])
    log.info('Clean interface Top-1 %.3f / original frozen Backbone %.3f',
             baseline['top1'],baseline['original_top1'])
    detail,summary=[],[]
    for kind in CHANNELS:
        for mode in (('none',) if kind=='awgn' else modes):
            for snr in snrs:
                reps=[]
                for ri in range(repeats):
                    seed=channel_seed(cfg['training']['seed'],kind,snr,ri)
                    vals=evaluate(model,val,device,ch,kind,mode,snr,seed,amp)
                    detail.append(dict(channel=kind,mode=mode,snr_db=snr,
                                       repeat=ri+1,seed=seed,**vals))
                    reps.append(vals)
                line=dict(channel=kind,mode=mode,snr_db=snr,repeats=repeats)
                for metric in ('top1','top5','index_acc'):
                    line[f'{metric}_mean']=mean(x[metric] for x in reps)
                    line[f'{metric}_std']=stdev(x[metric] for x in reps) if repeats>1 else 0.
                summary.append(line)
                log.info('%-8s %-15s SNR=%+3g | top1 %.2f ± %.2f / index %.2f',
                         kind,mode,snr,line['top1_mean'],line['top1_std'],line['index_acc_mean'])
    write_csv(out/'results_detail.csv',detail)
    write_csv(out/'results_summary.csv',summary)
    for metric in ('top1','top5','index_acc'):
        rows=[]
        for kind in CHANNELS:
            for mode in (('none',) if kind=='awgn' else modes):
                row=dict(channel=kind,mode=mode)
                for snr in snrs:
                    found=next(x for x in summary if x['channel']==kind and
                               x['mode']==mode and x['snr_db']==snr)
                    row[f'{snr:g}dB']=round(found[f'{metric}_mean'],4)
                rows.append(row)
        write_csv(out/f'{metric}_table.csv',rows)
    log.info('Saved result tables to %s',out)


def load_system(path,device):
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    model=System(ckpt['backbone'],ckpt['config'])
    model.load_state_dict(ckpt['model'],strict=True)
    model.to(device).eval()
    return model,ckpt


def main():
    p=argparse.ArgumentParser(description='Test independently saved model checkpoint')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--snrs',nargs='+',type=float)
    p.add_argument('--modes',nargs='+',choices=MODES)
    p.add_argument('--repeats',type=int)
    p.add_argument('--batch-size',type=int)
    p.add_argument('--workers',type=int)
    args=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,ckpt=load_system(args.checkpoint,device)
    cfg=ckpt['config']
    seed_everything(cfg['training']['seed'])
    _,val,classes=loaders(args.data_root,args.batch_size or cfg['training']['batch_size'],
                           cfg['training']['workers'] if args.workers is None else args.workers,train=False)
    if classes != ckpt['classes']:raise ValueError('Dataset classes/order differ from checkpoint')
    test_grid(model,val,device,cfg,args.out,logger(args.out),args.snrs,args.modes,args.repeats,
              bool(cfg['training']['amp']) and device.type=='cuda')

if __name__=='__main__':main()
