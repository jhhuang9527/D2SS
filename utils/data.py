from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN=(0.485,0.456,0.406)
STD=(0.229,0.224,0.225)


def loaders(root, batch=256, workers=8, train=True):
    root = Path(root).expanduser()
    vtf = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                              transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    val_set = datasets.ImageFolder(root/'val',vtf)
    if len(val_set.classes)!=100:
        raise ValueError(f'Expected 100 classes, found {len(val_set.classes)}')
    opts = dict(batch_size=batch,num_workers=workers,
                pin_memory=torch.cuda.is_available(),persistent_workers=workers>0)
    val = DataLoader(val_set,shuffle=False,**opts)
    if not train:
        return None,val,val_set.classes
    ttf = transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    train_set = datasets.ImageFolder(root/'train',ttf)
    if train_set.class_to_idx != val_set.class_to_idx:
        raise ValueError('train and val class folder names/order do not match')
    train_generator = torch.Generator()
    tr = DataLoader(train_set,shuffle=True,generator=train_generator,**opts)
    return tr,val,val_set.classes
