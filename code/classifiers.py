import torch

class OneLayerLinear(torch.nn.Module):
    def __init__(self, input_feature_num) -> None:
        super().__init__()
        
        self.linear = torch.nn.Linear(
            in_features=input_feature_num,
            out_features=1
        )
        
    def forward(self, x):
        x = self.linear(x)
        return x