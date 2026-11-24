import math
import torch
from torch.nn import functional as F


def epoch(model, loader, device, weights, stage='clean', optimizer=None,
          scaler=None, perturb=None, val_kind=None, seed=42,
          amp=False, grad_clip=1.0, phase='hard', val_radius=1.0):
    training = optimizer is not None
    model.train(training)
    generator = None if training else torch.Generator(device=device.type).manual_seed(int(seed))
    totals = dict(loss=0., ce=0., kd=0., top1=0., vq=0.,
                  top5=0, index_acc=0, count=0, token_spread=0., teacher_top1=0)
    codebook_size = model.codec.vq.k
    hist = torch.zeros(codebook_size, dtype=torch.long)
    predictions = torch.zeros(100, dtype=torch.long)
    quantized = False

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp):
                feats, teacher = model.features(images)
                if phase == 'continuous':
                    z = model.codec.encoder(feats)
                    recovered = model.codec.decoder(z)
                    q, prob = None, None
                elif phase == 'hard':
                    tx, q = model.codec.encode(feats)
                    if stage == 'clean':
                        rx = tx
                    elif val_kind == 'g2':
                        rx = perturb.g2(tx, generator=generator, radius=val_radius)
                    else:
                        rx = perturb(tx, generator=generator)
                    recovered, prob, _ = model.codec.decode(rx, q)
                else:
                    raise ValueError(f'Unknown curriculum phase {phase}')
                logits = model.classify(recovered)
                ce = F.cross_entropy(logits, labels)
                t = 2.
                kd = (F.kl_div(F.log_softmax(logits/t, dim=-1),
                               F.softmax(teacher/t, dim=-1),
                               reduction='batchmean') * t*t)
                vq = q['vq_loss'] if q is not None else ce.new_zeros(())

                if stage == 'clean':
                    loss = weights['ce']*ce + weights['kd']*kd + weights['vq']*vq
                else:
                    loss = (weights['perturbed_ce']*ce + weights['perturbed_kd']*kd
                            + weights['vq']*vq)
            if training:
                if scaler is None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.codec.parameters(), grad_clip)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.codec.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()

        count = labels.numel()
        hits = logits.detach().topk(5, dim=-1).indices.eq(labels[:, None])
        totals['top1'] += int(hits[:, 0].sum())
        totals['top5'] += int(hits.any(dim=-1).sum())
        totals['count'] += count
        totals['teacher_top1'] += int((teacher.detach().argmax(-1)==labels).sum())
        predictions += torch.bincount(logits.detach().argmax(-1).cpu(), minlength=100)
        if q is not None:
            quantized = True
            z = q['token'].detach().float()
            if z.size(0) >= 2:
                totals['token_spread'] += float(z.std(dim=0, unbiased=False).mean())*count
            hist += torch.bincount(q['indices'].detach().reshape(-1).cpu(),
                                   minlength=codebook_size)
            if phase == 'hard':
                totals['index_acc'] += int((prob.detach().argmax(-1)
                                            == q['indices']).sum())
        else:
            z = z.detach().float()
            if z.size(0) >= 2:
                totals['token_spread'] += float(z.std(dim=0, unbiased=False).mean())*count
        for name, value in (('loss',loss),('ce',ce),('kd',kd),('vq',vq)):
            totals[name] += float(value.detach())*count

    count = totals.pop('count')
    if not count:
        raise RuntimeError('Empty data loader')
    for name in ('loss','ce','kd','vq'):
        totals[name] /= count
    for name in ('top1','top5'):
        totals[name] = 100*totals[name]/count
    totals['token_spread'] /= count
    totals['teacher_top1'] = 100*totals['teacher_top1']/count
    totals['prediction_classes'] = int((predictions>0).sum())
    if quantized:
        frequency = hist.float()
        frequency /= frequency.sum().clamp_min(1)
        totals['active_ratio'] = float((hist>0).float().mean())
        totals['active_count'] = int((hist>0).sum())
        totals['perplexity'] = float(torch.exp(-(frequency*frequency.clamp_min(1e-12).log()).sum()))
    else:
        totals['active_ratio'] = float('nan')
        totals['active_count'] = 0
        totals['perplexity'] = float('nan')
    totals['index_acc'] = (100*totals['index_acc']/(count*model.codec.encoder.queries.shape[1])
                           if phase == 'hard' else float('nan'))
    return totals
