from collections import OrderedDict
import torch
from torch import nn

BACKBONES = ('resnet18', 'mobilenet_wd2', 'diracnet18v2')

class Backbone(nn.Module):
    def __init__(self, name):
        super().__init__()
        if name not in BACKBONES:
            raise ValueError(f'Unknown backbone: {name}')
        from pytorchcv.model_provider import get_model
        self.name = name
        self.net = get_model(name, pretrained=False)
        self.net.output = nn.Linear(self.net.output.in_features, 100)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def front(self, x):
        f = self.net.features
        x = f.init_block(x)
        x = f.stage1(x)
        if self.name == 'diracnet18v2':
            for name, block in f.stage2.named_children():
                if name == 'pool2':
                    break
                x = block(x)
        else:
            x = f.stage2(x)
        if self.name == 'mobilenet_wd2':
            x = f.stage3(x)
        if x.shape[1:] != (128, 28, 28):
            raise RuntimeError(f'Invalid split feature {tuple(x.shape)}, model={self.name}')
        return x

    def back(self, x):
        f = self.net.features
        if self.name == 'diracnet18v2':
            x = f.stage2.pool2(x)
            x = f.stage3(x)
            x = f.stage4(x)
            x = f.final_activ(x)
        elif self.name == 'mobilenet_wd2':
            x = f.stage4(x)
            x = f.stage5(x)
            if hasattr(f, 'stage6'):
                x = f.stage6(x)
        else:
            x = f.stage3(x)
            x = f.stage4(x)
        x = f.final_pool(x)
        return self.net.output(x.flatten(1))

    def forward(self, images):
        return self.back(self.front(images))

    def load_finetuned(self, path, classes=None):
        data = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(data, dict):
            raise ValueError('Checkpoint must be a dict')
        name = data.get('model_name')
        if name and name != self.name:
            raise ValueError(f'Weights are for {name}, expected {self.name}')
        if classes is not None and data.get('classes') is not None:
            if list(data['classes']) != list(classes):
                raise ValueError('ImageNet-100 class mapping differs from checkpoint')
        state = next((data[key] for key in ('model', 'state_dict', 'model_state_dict')
                      if isinstance(data.get(key), dict)), None)
        if state is None and all(isinstance(v, torch.Tensor) for v in data.values()):
            state = data
        if state is None:
            raise ValueError('No model state_dict in checkpoint')
        clean = OrderedDict()
        for k, v in state.items():
            while k.startswith('module.') or k.startswith('_orig_mod.'):
                k = k.split('.', 1)[1]
            if k.startswith('net.'):
                k = k[4:]
            clean[k] = v
        self.net.load_state_dict(clean, strict=True)
        self.requires_grad_(False)
        self.eval()
        return data
