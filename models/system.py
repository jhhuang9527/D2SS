import torch
from torch import nn
from .backbone import Backbone
from .codec import Codec


class System(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        self.backbone = Backbone(backbone)
        c = config['semantic']
        self.codec = Codec(n=c['n'], d=c['d'], k=c['k'], m=c['m'],
                           width=c['hidden_dim'], power=c['power'],
                           det_temp=c['det_temperature'], commitment=c['commitment_beta'],
                           grad_bridge=c['grad_bridge'],
                           signature_seed=c['signature_seed'])

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def features(self, images):
        with torch.no_grad():
            f = self.backbone.front(images)
            teacher = self.backbone.back(f)
        return f, teacher

    def classify(self, f):
        return self.backbone.back(f)

    def forward(self, images, corrupt=None, phase='hard'):
        f, teacher = self.features(images)
        recovered, q, prob = self.codec.forward_curriculum(f, phase, corrupt)
        logits = self.classify(recovered)
        return logits, teacher, q, prob
