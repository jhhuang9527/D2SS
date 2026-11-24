import math
import torch

MODES = ('none','imperfect_ls','imperfect_mmse','perfect_ls','perfect_mmse')
CHANNELS = ('awgn','rician','rayleigh')


def complex_normal(shape, device, generator, variance=1.):
    re = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    im = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    return torch.complex(re, im) * math.sqrt(variance / 2.)


class Channel:
    def __init__(self, power=1., rician_k=5., csi_error_var=0.01):
        if not (power > 0 and rician_k >= 0 and csi_error_var >= 0):
            raise ValueError('Invalid channel power, Rician K, or CSI error variance')
        self.power = float(power)
        self.rician_k = float(rician_k)
        self.csi_error_var = float(csi_error_var)

    def __call__(self, tx, kind, snr_db, mode, noise_gen, fading_gen, csi_gen):
        if kind not in CHANNELS or mode not in MODES:
            raise ValueError(f'Invalid channel/mode: {kind}/{mode}')
        if kind == 'awgn' and mode != 'none':
            raise ValueError('AWGN has no fading equalization')
        if tx.ndim != 3 or not torch.is_complex(tx):
            raise ValueError('Expected complex transmit tensor [B,N,M]')
        with torch.autocast(device_type=tx.device.type, enabled=False):
            b = tx.shape[0]
            n0 = self.power * 10.0 ** (-float(snr_db) / 10.)
            noise = complex_normal(tx.shape, tx.device, noise_gen, n0)
            if kind == 'awgn':
                return tx + noise
            h = complex_normal((b, 1, 1), tx.device, fading_gen)
            if kind == 'rician':
                k = self.rician_k
                h = math.sqrt(k / (k+1)) + h / math.sqrt(k+1)
            y = h * tx + noise
            if mode == 'none':
                return y
            if mode.startswith('imperfect'):
                h_hat = h + complex_normal(h.shape, tx.device, csi_gen, self.csi_error_var)
            else:
                h_hat = h
            power = h_hat.abs().square()
            if mode.endswith('_ls'):
                return y * h_hat.conj() / power.clamp_min(1e-8)
            return y * h_hat.conj() / (power + n0/self.power)
