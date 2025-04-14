from typing import Optional

import torch
from torch import nn, cat


class LazyMLP(nn.Sequential):
    def __init__(
            self,
            hidden_channels: list[int],
            activation: type[nn.Module] = nn.ReLU(),
            dropout: float = 0.0,
            pool: Optional[nn.Module] = None,  # nn.AdaptiveMaxPool1d(output_size=1),
    ):
        if not activation:
            activation = nn.Identity()

        layers = []
        for hidden_dim in hidden_channels:
            layers.append(nn.LazyLinear(out_features=hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
        super().__init__(*layers)

        if pool is True:
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.pool = pool

    def forward(self, x):
        x = x.float()  # Ensure the input is of type Float
        x = super().forward(x)
        if self.pool and len(x.shape) > 2:
            x = x.permute(0, 2, 1)
            x = self.pool(x).squeeze(-1)  # [batch_size, hidden_channels[-1]]
        return x


class SimpleConcatMLP(LazyMLP):
    def forward(self, *inputs):
        inputs = [x.mean(dim=1) if x.dim() > 2 else x for x in inputs]
        x = cat(inputs, dim=1)
        x = super().forward(x)
        return x


class ConcatMLP(nn.Module):
    def __init__(
            self,
            proj_dim: int,
            hidden_channels: list[int],
            activation: Optional[type[nn.Module]] = nn.ReLU(),
            dropout: float = 0.0,
    ):
        """
        Args:
            num_branches: Number of input branches.
            proj_dim: Output dimension for each branch (each branch is projected to this fixed size).
            hidden_channels: List of integers, defining the hidden layer sizes of the head MLP.
            activation: Activation module type to use (default: nn.ReLU).
            dropout: Dropout probability for both branch projections and head.
        """
        super().__init__()
        # Build a ModuleList of branch processors.
        self.branches = nn.ModuleList()
        self.head = LazyMLP(
            hidden_channels=hidden_channels,
            activation=activation,
            dropout=dropout
        )
        self.proj_dim = proj_dim
        self.activation = activation
        self.dropout = dropout

    def forward(self, *inputs):
        """
        For any input of higher dimensionality (e.g. [batch, sequence, features]), it takes the mean over the sequence dimension.
        """
        if not self.branches:
            self.branches = nn.ModuleList([
                LazyMLP(
                    hidden_channels=[self.proj_dim],
                    activation=self.activation,
                    dropout=self.dropout,
                ) for _ in range(len(inputs))
            ])

        branch_outs = []
        for branch, x in zip(self.branches, inputs):
            # If input is, e.g., [batch, sequence, features], average over the sequence dimension (dim=1).
            if x.dim() > 2:
                x = x.mean(dim=1)
            branch_outs.append(branch(x))
        # Concatenate along the feature dimension.
        concat = cat(branch_outs, dim=1)
        out = self.head(concat)
        return out


class KroneckerMLP(nn.Module):
    def __init__(self, alpha=1e-2, dropout=0.2):
        super().__init__()
        self.prot_comp_mix = nn.Sequential(
            nn.Dropout(dropout),
            nn.LazyLinear(1024),
            nn.LeakyReLU(alpha),
            nn.Dropout(dropout)
        )
        self.fc = nn.Sequential(
            nn.LazyLinear(512),
            nn.LeakyReLU(alpha),
            nn.Dropout(dropout),
        )

    def forward(self, compound, protein):
        comp_out = torch.cat((compound, torch.ones(compound.shape[0], 1, device=compound.device)), dim=1)
        prot_out = torch.cat((protein, torch.ones(protein.shape[0], 1, device=protein.device)), dim=1)
        output = torch.bmm(prot_out.unsqueeze(2), comp_out.unsqueeze(1)).flatten(start_dim=1)
        output = self.prot_comp_mix(output)
        output = torch.cat((output, prot_out, comp_out), 1)
        output = self.fc(output)
        return output


class ConcatDropoutMLP(nn.Sequential):
    def __init__(
            self,
            hidden_channels: list[int],
            activation: type[nn.Module] = nn.ReLU(),
            dropout: float = 0.5,
            pool: Optional[nn.Module] = None,  # nn.AdaptiveMaxPool1d(output_size=1),
    ):
        if not activation:
            activation = nn.Identity()

        layers = []
        for hidden_dim in hidden_channels:
            layers.append(nn.LazyLinear(out_features=hidden_dim))
            layers.append(activation)
        super().__init__(*layers)

        if pool is True:
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.pool = pool

        self.dropout = nn.Dropout(dropout)

    def forward(self, *inputs):
        inputs = [x.mean(dim=1) if x.dim() > 2 else x for x in inputs]
        x = cat(inputs, dim=1)
        x = self.dropout(x)
        x = x.float()  # Ensure the input is of type Float
        x = super().forward(x)
        if self.pool and len(x.shape) > 2:
            x = x.permute(0, 2, 1)
            x = self.pool(x).squeeze(-1)  # [batch_size, hidden_channels[-1]]
        return x


# class ConcatMLP(LazyMLP):
#     def forward(self, *inputs):
#         x = cat([*inputs], 1)
#         x = super().forward(x)
#         return x


# class ConcatMLP(MLP1):
#     def forward(self, *inputs):
#         x = cat([*inputs], 1)
#         for module in self:
#             x = module(x)
#         return x

