import torch.nn as nn

class BA3MotifGLNN(nn.Module):
    def __init__(self, device, num_unit, in_features, out_features):
        super().__init__()
        self.num_unit = num_unit
        self.node_emb = nn.Sequential(
            nn.Linear(in_features, 64), # node embedding layer
            nn.ReLU(),
        ).to(device)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU()
            ) for _ in range(num_unit)
        ]).to(device)
        self.lin1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        ).to(device)
        self.lin2 = nn.Sequential(
            nn.Linear(64, 3),
        ).to(device)

    def forward(self, x):
        x = self.node_emb(x)
        for i in range(self.num_unit):
            x = self.hidden_layers[i](x)
        x = self.lin1(x)
        output = self.lin2(x)
        return output
