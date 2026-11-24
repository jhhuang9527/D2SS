import torch
from torch.nn import functional as F


@torch.no_grad()
def kmeans_cosine(tokens, k, iters=12, seed=42):
    if tokens.ndim != 2 or tokens.size(0) < k:
        raise ValueError(f'Need >=K collected tokens; received {tokens.shape}')
    x = F.normalize(tokens.float(), dim=-1)
    rnd = torch.Generator(device=x.device.type).manual_seed(seed)
    n, d = x.shape
    centers = torch.empty(k, d, device=x.device)
    centers[0] = x[torch.randint(n, (1,), generator=rnd, device=x.device).item()]
    distances = (1 - x @ centers[0]).clamp_min(0)
    for i in range(1, k):
        weights = distances.square()
        if float(weights.sum()) < 1e-10:
            idx = torch.randint(n, (1,), generator=rnd, device=x.device).item()
        else:
            idx = torch.multinomial(weights, 1, generator=rnd).item()
        centers[i] = x[idx]
        distances = torch.minimum(distances, (1 - x @ centers[i]).clamp_min(0))
    for _ in range(iters):
        labels = (x @ centers.T).argmax(-1)
        counts = torch.bincount(labels, minlength=k)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, x)
        new_centers = F.normalize(sums / counts.clamp_min(1)[:, None], dim=-1)
        if (counts == 0).any():
            # Prefer poorly represented tokens for empty cluster replacements.
            difficult = (x * centers[labels]).sum(-1).argsort()
            missing = (counts == 0).nonzero(as_tuple=False).flatten()
            for j, code_id in enumerate(missing):
                new_centers[code_id] = x[difficult[j]]
        centers = new_centers
    labels = (x @ centers.T).argmax(-1)
    counts = torch.bincount(labels, minlength=k)
    return centers, counts


@torch.no_grad()
def initialize_simvq(system, loader, device, max_tokens=8192, iters=12,
                     seed=42, log=None):
    model = system
    model.eval()
    target = max(max_tokens, model.codec.vq.k)
    batch_tokens = []
    seen = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        f = model.backbone.front(images)
        z = model.codec.encoder(f).float().reshape(-1, model.codec.vq.dim)
        needed = target - seen
        batch_tokens.append(z[:needed].cpu())
        seen += min(len(z), needed)
        if seen >= target:
            break
    if seen < model.codec.vq.k:
        raise ValueError(f'K-means initialization needs >= {model.codec.vq.k} '
                         f'Encoder tokens; only {seen} were collected')
    data = torch.cat(batch_tokens, dim=0).to(device)
    spread = float(data.std(dim=0, unbiased=False).mean())
    centers, counts = kmeans_cosine(data, model.codec.vq.k, iters, seed)
    model.codec.vq.init_from_centers(centers)
    active = int((counts > 0).sum())
    if log is not None:
        log.info('K-means SimVQ initialized: tokens=%d, cluster_active=%d/%d, '
                 'encoder_spread=%.5f, iters=%d',
                 seen, active, model.codec.vq.k, spread, iters)
        if spread < 0.02 or active < 0.5*model.codec.vq.k:
            log.warning('Low encoder diversity before VQ. Check continuous '
                        'classification accuracy before starting full training.')
    return {'tokens':seen, 'active_clusters':active, 'encoder_spread':spread}
