import torch
from torch import nn, rand
from torch.autograd import Variable
from typing import Callable, Optional, List


class LazyCNN(nn.Module):
    def __init__(
            self,
            num_features: int,
            embed_dim: Optional[int],
            filters: List[int],
            kernels: List[int],
            activation: Optional[nn.Module] = nn.ReLU(),
            pool: Optional[nn.Module] = None,  # nn.AdaptiveMaxPool1d(output_size=1),
            out_channels: Optional[int] = None,
            # fc: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.embedding = None
        self.fc = None
        if not activation:
            activation = nn.Identity()

        if pool is True:
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.pool = pool

        if embed_dim:
            self.embedding = nn.Embedding(num_features, embed_dim)

        layers = []
        for i in range(len(filters)):
            layers.append(nn.LazyConv1d(out_channels=filters[i], kernel_size=kernels[i], padding='same'))
            layers.append(activation)

        self.conv = nn.Sequential(*layers)
        if out_channels:
            self.fc = nn.LazyLinear(out_channels)

    def forward(self, v):
        v_lengths = None
        if hasattr(v, 'lengths'):
            v_lengths = v.lengths

        if self.embedding:
            v = self.embedding(v.long())

        v = v.permute(0, 2, 1)
        v = self.conv(v)

        if self.pool:
            v = self.pool(v).squeeze(-1)  # [batch_size, num_features]
        else:
            v = v.permute(0, 2, 1)  # [batch_size, length, num_features]

        if self.fc:
            v = v.view(v.size(0), -1)
            v = self.fc(v)  # [batch_size, out_channels]

        if v_lengths is not None:
            v.lengths = v_lengths

        return v


class PretrainOnLazyCNN(LazyCNN):
    def __init__(self, pretrain, freeze_layers, **kwargs):
        super().__init__(**kwargs)
        self.pretrain = pretrain

        # If freeze_layers is None, do not make any change
        if freeze_layers is None:
            pass
        # If freeze_layers is True, freeze all layers
        elif freeze_layers is True:
            for param in self.pretrain.parameters():
                param.requires_grad = False
        # If freeze_layers is False, do not freeze any layers
        elif freeze_layers is False:
            for param in self.pretrain.parameters():
                param.requires_grad = True
        # If freeze_layers is an integer, freeze the last N layers
        elif isinstance(freeze_layers, int):
            total_layers = len(list(self.pretrain.parameters()))
            # Freeze the last `freeze_layers` layers
            for i, param in enumerate(self.pretrain.parameters()):
                if i >= total_layers - freeze_layers:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def forward(self, v):
        v = self.pretrain(v)[0]  # 0th being the unpooled output for most PLM models
        v = super().forward(v)

        return v


class CNN1D(nn.Module):
    """Used in GraphDTA"""
    def __init__(
            self,
            num_features: int,
            embed_dim: int,
            filter: int,
            kernel: int,
            out_features: int,
    ):
        super().__init__()

        self.embedding = nn.Embedding(num_features, embed_dim)
        self.conv = nn.LazyConv1d(out_channels=filter, kernel_size=kernel)
        self.fc = nn.LazyLinear(out_features)

    def forward(self, v):
        v = self.embedding(v.long())
        v = self.conv(v)
        v = torch.flatten(v, start_dim=1)
        v = self.fc(v)
        return v


class CNN(nn.Sequential):
    def __init__(
            self,
            filters: list[int],
            kernels: list[int],
            max_sequence_length: int,
            in_channels: int,
            out_channels: int
    ):
        super().__init__()
        num_layer = len(filters)
        channels = [in_channels] + filters
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=channels[i],
                                             out_channels=channels[i+1],
                                             kernel_size=kernels[i])
                                   for i in range(num_layer)])
        n_size = self._get_conv_output((in_channels, max_sequence_length))
        self.fc1 = nn.Linear(n_size, out_channels)

    def _forward_features(self, x):
        for layer in self.conv:
            x = nn.functional.relu(layer(x))
        x = nn.functional.adaptive_max_pool1d(x, output_size=1)
        return x

    def _get_conv_output(self, shape):
        bs = 1
        input_feat = Variable(rand(bs, *shape))
        output_feat = self._forward_features(input_feat)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, v):
        v = self._forward_features(v.float())
        v = v.view(v.size(0), -1)
        v = self.fc1(v)
        return v
