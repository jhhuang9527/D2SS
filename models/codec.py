import math
import torch
from torch import nn
from torch.nn import functional as F


def gn(ch):
    return nn.GroupNorm(math.gcd(ch, 16), ch)


class DWBlock(nn.Module):
    def __init__(self, inp, out, down=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride=2 if down else 1, padding=1, groups=inp, bias=False),
            gn(inp), nn.SiLU(), nn.Conv2d(inp, out, 1, bias=False), gn(out), nn.SiLU(),
            nn.Conv2d(out, out, 3, padding=1, groups=out, bias=False),
            gn(out))
        self.skip = nn.Identity() if inp == out and not down else nn.Sequential(
            nn.Conv2d(inp, out, 1, stride=2 if down else 1, bias=False), gn(out))

    def forward(self, x):
        return F.silu(self.net(x) + self.skip(x))


class Encoder(nn.Module):
    def __init__(self, tokens, dim, width=256):
        super().__init__()
        if width % 4:
            raise ValueError('hidden_dim must be divisible by 4')
        self.stem = nn.Sequential(nn.Conv2d(128, 192, 1, bias=False), gn(192), nn.SiLU(),
                                  DWBlock(192, 192))
        self.down1 = DWBlock(192, width, down=True)
        self.down2 = DWBlock(width, width, down=True)

        coords = torch.arange(7, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        omega = torch.exp(-math.log(10000.0) * torch.arange(width // 4).float()
                          / max(1, width // 4 - 1))
        yp = yy.reshape(-1, 1) * omega[None]
        xp = xx.reshape(-1, 1) * omega[None]
        pos = torch.cat((yp.sin(), yp.cos(), xp.sin(), xp.cos()), dim=-1)
        self.register_buffer('pos', pos[None], persistent=False)

        self.queries = nn.Parameter(torch.empty(1, tokens, width))
        nn.init.orthogonal_(self.queries[0])
        with torch.no_grad():
            self.queries.mul_(0.5 * math.sqrt(width))
        self.cross = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.mix = nn.TransformerEncoderLayer(width, 4, width*2, dropout=0.,
                                               batch_first=True, norm_first=True)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim), nn.LayerNorm(dim))

    def forward(self, x):
        x = self.down2(self.down1(self.stem(x)))
        spatial = x.flatten(2).transpose(1, 2) + self.pos
        n = self.queries.shape[1]
        rows = max(1, int(math.sqrt(n)))
        cols = math.ceil(n / rows)

        anchors = F.adaptive_avg_pool2d(x, (rows, cols))
        anchors = anchors.flatten(2).transpose(1, 2)[:, :n]
        query = self.queries.expand(x.size(0), -1, -1) + anchors
        tokens, _ = self.cross(query, spatial, spatial, need_weights=False)
        tokens = self.norm(tokens + query)
        return self.out(self.mix(tokens))


class SimVQ(nn.Module):
    def __init__(self, k, dim, commitment=0.25):
        super().__init__()
        self.k, self.dim, self.commitment = k, dim, commitment
        self.register_buffer('basis', F.normalize(torch.randn(k, dim), dim=-1))
        self.transform = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim)*0.002)

    def codebook(self):
        with torch.autocast(device_type=self.basis.device.type, enabled=False):
            return F.normalize(self.basis.float() @ self.transform.float(), dim=-1) * math.sqrt(self.dim)

    @torch.no_grad()
    def init_from_centers(self, centers):
        if centers.shape != (self.k, self.dim):
            raise ValueError(f'Expected K-means centers [{self.k},{self.dim}]')
        self.basis.copy_(F.normalize(centers.float(), dim=-1).to(self.basis.device))
        self.transform.copy_(torch.eye(self.dim, device=self.transform.device))

    def forward(self, z):
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            codes = self.codebook()
            sim = F.normalize(z, dim=-1) @ F.normalize(codes, dim=-1).T
            indices = sim.argmax(dim=-1)
            selected = codes[indices]
            vq_loss = (F.mse_loss(selected, z.detach()) +
                       self.commitment * F.mse_loss(z, selected.detach()))
        return dict(indices=indices, token=z, vq_loss=vq_loss)


class Signatures(nn.Module):
    def __init__(self, m, k, seed=42, power=1., temp=0.05):
        super().__init__()
        if not (k >= 2):
            raise ValueError('Random dictionary uses K >= 2 by default')
        if power <= 0 or temp <= 0:
            raise ValueError('Transmit power and detector temperature must be positive')
        self.m, self.k, self.power, self.temp = m, k, power, temp
        rng = torch.Generator(device='cpu').manual_seed(int(seed))
        re = torch.randn(m, k, generator=rng)
        im = torch.randn(m, k, generator=rng)
        v = torch.complex(re, im) * (2.0**-0.5)
        self.register_buffer('v', F.normalize(v, dim=0).to(torch.complex64))

    def transmit(self, indices):
        with torch.autocast(device_type=indices.device.type, enabled=False):
            return self.v.T[indices] * math.sqrt(self.m * self.power)

    def detect(self, received):
        with torch.autocast(device_type=received.device.type, enabled=False):
            corr = received.to(torch.complex64) @ self.v.conj()
            energy = corr.abs().square()
            normalized = energy / energy.sum(-1, keepdim=True).clamp_min(1e-12)
            probs = F.softmax(normalized / self.temp, dim=-1)
            return probs, normalized.argmax(-1), normalized

    def correlation_stats(self):
        gram = self.v.conj().T @ self.v
        eye = torch.eye(self.k, dtype=gram.dtype, device=gram.device)
        mask = ~torch.eye(self.k, dtype=torch.bool, device=gram.device)
        return dict(mean_offdiag_energy=float(((gram-eye).abs().square())[mask].mean()),
                    max_offdiag_corr=float(gram.abs()[mask].max()),
                    max_unit_norm_error=float((self.v.abs().square().sum(0)-1).abs().max()))


class Refiner(nn.Module):
    def __init__(self, dim, k, width=256, layers=2):
        super().__init__()
        self.k = k
        self.in_proj = nn.Linear(dim, width)
        self.reliability = nn.Linear(3, width)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(width, 4, width*2, dropout=0.,
                                       batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.out_proj = nn.Linear(width, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, tokens, prob):
        top2 = prob.topk(min(2, self.k), dim=-1).values
        peak = top2[..., :1]
        margin = (top2[..., :1] - top2[..., 1:2]) if self.k > 1 else peak
        entropy = -(prob.clamp_min(1e-8)*prob.clamp_min(1e-8).log()).sum(-1, keepdim=True)/math.log(self.k)
        inp = self.in_proj(tokens) + self.reliability(torch.cat((peak, margin, entropy), dim=-1))
        for layer in self.layers:
            inp = layer(inp)
        return tokens + self.out_proj(inp)


class Decoder(nn.Module):
    def __init__(self, dim, width=256):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(dim, width), nn.LayerNorm(width))
        self.queries = nn.Parameter(torch.randn(1, 49, width) * 0.02)
        self.cross = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.mix = nn.TransformerEncoderLayer(width, 4, width*2, dropout=0.,
                                               batch_first=True, norm_first=True)
        self.up1 = DWBlock(width, 192)
        self.up2 = DWBlock(192, 128)
        self.out = nn.Conv2d(128, 128, 1)

    def forward(self, z):
        memory = self.in_proj(z)
        query = self.queries.expand(z.size(0), -1, -1)
        x, _ = self.cross(query, memory, memory, need_weights=False)
        x = self.mix(self.norm(x + query))
        x = x.transpose(1, 2).reshape(z.size(0), -1, 7, 7)
        x = self.up1(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))
        x = self.up2(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))
        return self.out(x)


class Codec(nn.Module):
    def __init__(self, n=8, d=256, k=128, m=196, width=256,
                 power=1., det_temp=0.05, commitment=0.25,
                 grad_bridge=1.0, signature_seed=42):
        super().__init__()
        self.encoder = Encoder(n, d, width)
        self.vq = SimVQ(k, d, commitment)
        self.sig = Signatures(m, k, signature_seed, power, det_temp)
        self.refiner = Refiner(d, k, width, layers=2)
        self.decoder = Decoder(d, width)
        self.grad_bridge = float(grad_bridge)

    def encode(self, feature):
        q = self.vq(self.encoder(feature))
        return self.sig.transmit(q['indices']), q

    def decode(self, received, q=None):
        probs, detected_indices, normalized_energy = self.sig.detect(received)

        tokens = self.vq.codebook()[detected_indices]
        if self.training and q is not None and self.grad_bridge:
            z = q['token']
            tokens = tokens + self.grad_bridge*(z-z.detach())
        tokens = self.refiner(tokens, probs)
        return self.decoder(tokens), probs, normalized_energy

    def forward_curriculum(self, feature, phase='hard', corrupt=None):
        if phase == 'continuous':
            return self.decoder(self.encoder(feature)), None, None
        if phase != 'hard':
            raise ValueError('Only continuous warmup and VQ are supported')
        return self.forward(feature, corrupt)

    def forward(self, feature, corrupt=None):
        tx,q = self.encode(feature)
        rx = tx if corrupt is None else corrupt(tx)
        recovered, probs, _ = self.decode(rx, q)
        return recovered, q, probs
