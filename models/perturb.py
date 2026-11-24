import torch


class Perturb:
    def __init__(self, v, radii):
        if not radii or any(r < 0 for r in radii):
            raise ValueError('Need nonnegative non-empty perturbation radii')
        self.v, self.radii = v, radii

    def g2(self, tx, generator=None, radius=None):
        b,n,m = tx.shape
        k = self.v.shape[1]
        re = torch.randn((b,n,k),device=tx.device,generator=generator)
        im = torch.randn((b,n,k),device=tx.device,generator=generator)
        xi = torch.complex(re,im) * (2.0**-0.5)
        delta = xi @ self.v.T
        if radius is None:
            choices = torch.tensor(self.radii,device=tx.device)
            idx = torch.randint(len(choices),(b,),device=tx.device,generator=generator)
            radius = choices[idx]
        if not isinstance(radius,torch.Tensor):
            radius = torch.full((b,),float(radius),device=tx.device)
        tx_norm = tx.detach().abs().square().sum((1,2),keepdim=True).sqrt()
        delta_norm = delta.abs().square().sum((1,2),keepdim=True).sqrt().clamp_min(1e-12)
        return tx + delta*(radius.view(b,1,1)*tx_norm/delta_norm)

    def __call__(self,tx,generator=None):
        return self.g2(tx,generator)
