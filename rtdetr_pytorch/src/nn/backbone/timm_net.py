'''by heikki.huttunen@visy.fi
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from src.core import register
import timm

__all__ = ['TimmNet']

@register
class TimmNet(nn.Module):
    
    def __init__(
            self, 
            name,
            return_idx=[0, 1, 2, 3],
            out_indices = None,
            freeze_at=-1, 
            freeze_norm=True, 
            pretrained=False):
        
        super().__init__()

        self.model = timm.create_model(name, features_only=True, out_indices=out_indices, pretrained=pretrained)
        
        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)
                
    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        
        return self.model(x)


