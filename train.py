import argparse
from pathlib import Path
import torch
from models import System, Perturb
from utils.cluster import initialize_simvq
from utils.data import loaders
from utils.engine import epoch as run_epoch
from utils.misc import (config, logger, seed_everything, restore_rng, save,
                        state_for_save, write_csv, snapshot_config)
from test import test_grid


def parse_args():
    p = argparse.ArgumentParser(description='D2SS')
    p.add_argument('--backbone', required=True, choices=('resnet18','mobilenet_wd2','diracnet18v2'))
    p.add_argument('--weights', help='Existing ImageNet-100 classification best.pt; new run only')
    p.add_argument('--data-root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--resume', default=None, help='Resume from a checkpoint produced by this project')
    for name in ('n','d','k','m'):
        p.add_argument('--'+name, type=int)
    for name in ('batch-size','workers','stage1-epochs','stage2-epochs',
                 'warmup-epochs','kmeans-tokens','kmeans-iters',
                 'test-repeats'):
        p.add_argument('--'+name, type=int)
    p.add_argument('--no-final-test', action='store_true')
    return p.parse_args()


def optimizer_for(model, cfg, stage):
    lr = cfg['training']['lr_stage1'] if stage == 'clean' else cfg['training']['lr_stage2']
    opt = torch.optim.AdamW(model.codec.parameters(), lr=lr,
                            weight_decay=cfg['training']['weight_decay'])
    n = (cfg['training']['stage1_epochs'] if stage == 'clean'
         else cfg['training']['stage2_epochs'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n, eta_min=lr*.03)
    return opt, sched


def curriculum_limits(pconf):
    """Continuous warmup then VQ, guaranteeing at least one hard epoch."""
    epochs = pconf['stage1_epochs']
    return min(pconf['warmup_epochs'], max(0, (epochs-1)//3)) if epochs < 20 else min(pconf['warmup_epochs'], epochs-1)


def phase_for_epoch(epoch, warmup):
    return 'continuous' if epoch <= warmup else 'hard'


def main():
    args = parse_args()
    if not args.resume and not args.weights:
        raise ValueError('New experiment needs --weights')
    c = config(args.config)
    for name in ('n','d','k','m'):
        val = getattr(args, name)
        if val is not None:
            c['semantic'][name] = val
    for flag, name in (('batch_size','batch_size'), ('workers','workers'),
                       ('stage1_epochs','stage1_epochs'), ('stage2_epochs','stage2_epochs'),
                       ('warmup_epochs','warmup_epochs'),
                       ('kmeans_tokens','kmeans_tokens'), ('kmeans_iters','kmeans_iters')):
        value = getattr(args, flag)
        if value is not None:
            c['training'][name] = value
    if args.test_repeats is not None:
        c['test']['repeats'] = args.test_repeats
    pconf = c['training']
    if any(pconf[name] < 1 for name in ('stage1_epochs','stage2_epochs','batch_size',
                                      'eval_every','kmeans_iters')):
        raise ValueError('Epochs/batch size/eval interval/K-means iterations must be >=1')
    if pconf['workers'] < 0 or pconf['warmup_epochs'] < 0:
        raise ValueError('Workers and curriculum durations must be nonnegative')
    warm = curriculum_limits(pconf)
    out = Path(args.out).expanduser()
    if not args.resume and (out/'last.pt').exists():
        raise FileExistsError(f'Existing checkpoint at {out}/last.pt; use --resume or a new --out')
    log = logger(out)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = pconf['amp'] and device.type == 'cuda'
    seed_everything(pconf['seed'])
    train, val, classes = loaders(args.data_root, pconf['batch_size'], pconf['workers'])
    model = System(args.backbone, c)
    ckpt = None
    vq_initialized = False
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        if ckpt.get('format_version') != 5:
            raise ValueError('Old checkpoints are incompatible; start a new output directory')
        if ckpt['config'] != c or ckpt['backbone'] != args.backbone or ckpt['classes'] != classes:
            raise ValueError('Resume settings/backbone/dataset class order differ')
        model.load_state_dict(ckpt['model'], strict=True)
        vq_initialized = bool(ckpt.get('vq_initialized', False))
        log.info('Resuming %s epoch %d phase=%s from %s',
                 ckpt['stage'], ckpt['epoch'], ckpt.get('phase'), args.resume)
    else:
        model.backbone.load_finetuned(args.weights, classes)
        log.info('Loaded fine-tuned ImageNet-100 %s from %s', args.backbone, args.weights)
    model.to(device)
    snapshot_config(out, dict(config=c, backbone=args.backbone, pretrained=args.weights,
                              stage1_effective=dict(continuous=warm,
                                                    hard=pconf['stage1_epochs']-warm)))
    perturb = Perturb(model.codec.sig.v, pconf['g2_radii'])
    stats = model.codec.sig.correlation_stats()
    log.info('Signal dictionary seed=%d, mean_offdiag_energy=%.6g, '
             'max_offdiag_corr=%.6g, norm_error=%.3g',
             c['semantic']['signature_seed'],
             stats['mean_offdiag_energy'], stats['max_offdiag_corr'],
             stats['max_unit_norm_error'])
    scaler = torch.amp.GradScaler('cuda') if amp else None
    log.info('Frozen %s N=%d D=%d K=%d M=%d uses=%d | curriculum=%d continuous + '
             '%d VQ, then %d robust training',
             args.backbone, *[c['semantic'][i] for i in ('n','d','k','m')],
             c['semantic']['n']*c['semantic']['m'], warm,
             pconf['stage1_epochs']-warm, pconf['stage2_epochs'])
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError('Backbone is not frozen')

    for stage in ('clean','robust'):
        if ckpt is not None and ckpt['stage'] == 'robust' and stage == 'clean':
            continue
        max_epochs = pconf['stage1_epochs'] if stage == 'clean' else pconf['stage2_epochs']
        opt, sched = optimizer_for(model, c, stage)
        start, best, best_epoch = 1, -1., 0
        if ckpt is not None and ckpt['stage'] == stage:
            opt.load_state_dict(ckpt['optimizer'])
            sched.load_state_dict(ckpt['scheduler'])
            if scaler is not None and ckpt.get('scaler') is not None:
                scaler.load_state_dict(ckpt['scaler'])
            start = ckpt['epoch']+1
            best, best_epoch = ckpt['best'], ckpt['best_epoch']
            restore_rng(ckpt['rng'])
        elif stage == 'robust':
            initial = out/'stage1_best.pt'
            if not initial.is_file():
                raise FileNotFoundError(f'Missing VQ Stage1 best: {initial}')
            initial_ckpt = torch.load(initial, map_location='cpu', weights_only=False)
            if initial_ckpt.get('phase') != 'hard':
                raise RuntimeError('Stage2 requires a real Stage1 best')
            model.load_state_dict(initial_ckpt['model'], strict=True)
            vq_initialized = True
            log.info('Stage2 initialized from Stage1 best epoch=%d',
                     initial_ckpt['epoch'])
        ckpt = None
        log.info('STAGE %s start=%d/%d', stage, start, max_epochs)
        for ep in range(start, max_epochs+1):
            phase = phase_for_epoch(ep, warm) if stage == 'clean' else 'hard'
            if phase != 'continuous' and not vq_initialized:
                log.info('Collecting train Encoder outputs for spherical K-means ...')
                train.generator.manual_seed(pconf['seed'] + 700000)
                initialize_simvq(model, train, device, pconf['kmeans_tokens'],
                                 pconf['kmeans_iters'], pconf['seed']+123, log)
                vq_initialized = True
                opt.state.pop(model.codec.vq.transform, None)
            model.train()
            epoch_seed = pconf['seed'] + ep + (0 if stage == 'clean' else 1000000)
            torch.manual_seed(epoch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(epoch_seed)
            train.generator.manual_seed(epoch_seed)
            lr = opt.param_groups[0]['lr']
            weights = pconf['loss_stage1'] if stage == 'clean' else pconf['loss_stage2']
            tr = run_epoch(model, train, device, weights, stage, optimizer=opt,
                           scaler=scaler, perturb=perturb, amp=amp,
                           grad_clip=pconf['grad_clip'], phase=phase)
            sched.step()
            record = dict(stage=stage, phase=phase, epoch=ep, lr=lr,
                          train_loss=tr['loss'], train_top1=tr['top1'],
                          train_top5=tr['top5'], teacher_top1=tr['teacher_top1'],
                          ce=tr['ce'], kd=tr['kd'], vq=tr['vq'],
                          train_index_acc=tr['index_acc'], active_ratio=tr['active_ratio'],
                          active_codes=tr['active_count'], perplexity=tr['perplexity'],
                          token_spread=tr['token_spread'], pred_classes=tr['prediction_classes'],
                          signature_seed=c['semantic']['signature_seed'],
                          val_clean='', val_g2='',
                          score='', best='', best_epoch='')
            if ep % pconf['eval_every'] == 0 or ep == max_epochs or (
                    stage == 'clean' and ep == warm+1):
                model.eval()
                if stage == 'clean':
                    vv = run_epoch(model, val, device, weights, 'clean', amp=amp,
                                   phase=phase)
                    record['val_clean'] = vv['top1']
                    score = vv['top1']
                    eligible = (phase == 'hard')
                else:
                    vg2 = run_epoch(model, val, device, weights, 'robust',
                                    perturb=perturb, val_kind='g2',
                                    seed=pconf['seed']+1234, amp=amp, phase='hard',
                                    val_radius=pconf['g2_val_radius'])
                    score = vg2['top1']
                    record['val_g2'] = vg2['top1']
                    eligible = True
                record['score'] = score
                if eligible and score > best:
                    best, best_epoch = score, ep
                    path = out/('stage1_best.pt' if stage == 'clean' else 'best.pt')
                    save(path, state_for_save(model,opt,sched,scaler,c,classes,args.backbone,
                                              stage,ep,best,best_epoch,
                                              phase=phase,vq_initialized=vq_initialized))
                    log.info('%s BEST epoch=%d phase=%s top1/score=%.2f%%',
                             stage,ep,phase,best)
            record['best'], record['best_epoch'] = best, best_epoch
            state = state_for_save(model,opt,sched,scaler,c,classes,args.backbone,
                                   stage,ep,best,best_epoch,
                                   phase=phase,vq_initialized=vq_initialized)
            save(out/'last.pt', state)
            if stage == 'clean' and ep == max_epochs:
                save(out/'stage1_last.pt', state)
            write_csv(out/'metrics.csv', [record], append=True)
            used = '--' if phase=='continuous' else f"{tr['active_count']}/{c['semantic']['k']}"
            idx = '--' if phase!='hard' else f"{tr['index_acc']:.1f}%"
            log.info('%-6s %3d/%d %-10s lr %.6g Top1 %.2f%% (teacher %.2f%%) '
                     'CE %.3f KD %.3f VQ %.3f idx %s used %s PPL %.1f '
                     'spread %.4f pred_classes %d best %.2f%%',
                     stage, ep, max_epochs, phase, lr, tr['top1'],
                     tr['teacher_top1'],tr['ce'],tr['kd'],tr['vq'],idx,used,
                     tr['perplexity'],tr['token_spread'],tr['prediction_classes'],best)
            if phase == 'hard' and stage == 'clean' and ep >= warm+5 and tr['active_count'] <= 1:
                log.warning('VQ still collapsed to one codeword in hard phase. '
                            'Inspect continuous accuracy and encoder_spread.')

    log.info('Finished; evaluating stage2 best.pt and last.pt on physical channels')
    if not args.no_final_test:
        for label in ('best','last'):
            saved = torch.load(out/f'{label}.pt', map_location='cpu', weights_only=False)
            if saved['stage'] != 'robust':
                raise RuntimeError(f'{label}.pt is not a robust Stage2 checkpoint')
            model.load_state_dict(saved['model'], strict=True)
            test_grid(model,val,device,c,out/f'test_{label}',log,amp=amp)
    log.info('Finished.')


if __name__ == '__main__':
    main()
