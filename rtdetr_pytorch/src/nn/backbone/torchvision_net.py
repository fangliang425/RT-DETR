'''by heikki.huttunen@visy.fi
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from src.core import register

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision

__all__ = ['TorchvisionNet']

@register
class TorchvisionNet(nn.Module):
    
    def __init__(
            self, 
            name,
            return_idx=[0, 1, 2, 3],
            out_names = None,
            freeze_at=-1, 
            freeze_norm=True, 
            pretrained=False):
        
        super().__init__()

        if name not in torchvision.models.list_models():
            print("Not")
        self.model = getattr(torchvision.models, name)(weights = "DEFAULT")
        train_nodes, eval_nodes = get_graph_node_names(self.model)

        for node_name in out_names:
            if node_name not in train_nodes:
                raise ValueError(f"Node {node_name} not found in {name}.")

        print(f"Creating feature extractor")        
        self.body = create_feature_extractor(self.model, return_nodes=out_names)
        print(f"Feature extractor created")        
        
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
        
        return self.body(x)


