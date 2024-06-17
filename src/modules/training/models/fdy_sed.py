# Some codes are adopted from https://github.com/DCASE-REPO/DESED_task
from collections.abc import Sequence

import torch
import torch.nn.functional as F


class GLU(torch.nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.linear = torch.nn.Linear(in_dim, in_dim)

    def forward(self, x):  # x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1))  # x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2)  # x size = [batch, chan, freq, frame]
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(torch.nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.sigmoid = torch.nn.Sigmoid()
        self.linear = torch.nn.Linear(in_dim, in_dim)

    def forward(self, x):  # x size = [batch, chan, freq, frame]
        lin = self.linear(x.permute(0, 2, 3, 1))  # x size = [batch, freq, frame, chan]
        lin = lin.permute(0, 3, 1, 2)  # x size = [batch, chan, freq, frame]
        sig = self.sigmoid(lin)
        res = x * sig
        return res


class Dynamic_conv2d(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, bias=False, n_basis_kernels=4, temperature=31, pool_dim="freq") -> None:
        super().__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_dim = pool_dim

        self.n_basis_kernels = n_basis_kernels
        self.attention = attention2d(in_planes, self.kernel_size, self.stride, self.padding, n_basis_kernels, temperature, pool_dim)

        self.weight = torch.nn.Parameter(torch.randn(n_basis_kernels, out_planes, in_planes, self.kernel_size, self.kernel_size), requires_grad=True)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(n_basis_kernels, out_planes))
        else:
            self.bias = None

        for i in range(self.n_basis_kernels):
            torch.nn.init.kaiming_normal_(self.weight[i])

    def forward(self, x):  # x size : [bs, in_chan, frames, freqs]
        if self.pool_dim in ["freq", "chan"]:
            softmax_attention = self.attention(x).unsqueeze(2).unsqueeze(4)  # size : [bs, n_ker, 1, frames, 1]
        elif self.pool_dim == "time":
            softmax_attention = self.attention(x).unsqueeze(2).unsqueeze(3)  # size : [bs, n_ker, 1, 1, freqs]
        elif self.pool_dim == "both":
            softmax_attention = self.attention(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # size : [bs, n_ker, 1, 1, 1]

        batch_size = x.size(0)

        aggregate_weight = self.weight.view(-1, self.in_planes, self.kernel_size, self.kernel_size)  # size : [n_ker * out_chan, in_chan]

        if self.bias is not None:
            aggregate_bias = self.bias.view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding)
            # output size : [bs, n_ker * out_chan, frames, freqs]

        output = output.view(batch_size, self.n_basis_kernels, self.out_planes, output.size(-2), output.size(-1))
        # output size : [bs, n_ker, out_chan, frames, freqs]

        if self.pool_dim in ["freq", "chan"]:
            assert softmax_attention.shape[-2] == output.shape[-2]
        elif self.pool_dim == "time":
            assert softmax_attention.shape[-1] == output.shape[-1]

        output = torch.sum(output * softmax_attention, dim=1)  # output size : [bs, out_chan, frames, freqs]

        return output


class attention2d(torch.nn.Module):
    def __init__(self, in_planes, kernel_size, stride, padding, n_basis_kernels, temperature, pool_dim) -> None:
        super().__init__()
        self.pool_dim = pool_dim
        self.temperature = temperature

        hidden_planes = int(in_planes / 4)

        if hidden_planes < 4:
            hidden_planes = 4

        if pool_dim != "both":
            self.conv1d1 = torch.nn.Conv1d(in_planes, hidden_planes, kernel_size, stride=stride, padding=padding, bias=False)
            self.bn = torch.nn.BatchNorm1d(hidden_planes)
            self.relu = torch.nn.ReLU(inplace=True)
            self.conv1d2 = torch.nn.Conv1d(hidden_planes, n_basis_kernels, 1, bias=True)
            for m in self.modules():
                if isinstance(m, torch.nn.Conv1d):
                    torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
                if isinstance(m, torch.nn.BatchNorm1d):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
        else:
            self.fc1 = torch.nn.Linear(in_planes, hidden_planes)
            self.relu = torch.nn.ReLU(inplace=True)
            self.fc2 = torch.nn.Linear(hidden_planes, n_basis_kernels)

    def forward(self, x):  # x size : [bs, chan, frames, freqs]
        if self.pool_dim == "freq":
            x = torch.mean(x, dim=3)  # x size : [bs, chan, frames]
        elif self.pool_dim == "time":
            x = torch.mean(x, dim=2)  # x size : [bs, chan, freqs]
        elif self.pool_dim == "both":
            # x = torch.mean(torch.mean(x, dim=2), dim=1)  #x size : [bs, chan]
            x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        elif self.pool_dim == "chan":
            x = torch.mean(x, dim=1)  # x size : [bs, freqs, frames]

        if self.pool_dim != "both":
            x = self.conv1d1(x)  # x size : [bs, hid_chan, frames]
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv1d2(x)  # x size : [bs, n_ker, frames]
        else:
            x = self.fc1(x)  # x size : [bs, hid_chan]
            x = self.relu(x)
            x = self.fc2(x)  # x size : [bs, n_ker]

        return F.softmax(x / self.temperature, 1)


class CNN(torch.nn.Module):
    def __init__(
        self,
        n_input_ch: int,
        activation: str = "Relu",
        conv_dropout: float = 0,
        kernel: Sequence[int] = [3, 3, 3],
        pad: Sequence[int] = [1, 1, 1],
        stride: Sequence[int] = [1, 1, 1],
        n_filt: Sequence[int] = [64, 64, 64],
        pooling: Sequence[int | tuple[int, int]] = [(1, 4), (1, 4), (1, 4)],
        normalization: str = "batch",
        n_basis_kernels: int = 4,
        DY_layers: Sequence[int] = [0, 1, 1, 1, 1, 1, 1],
        temperature: int = 31,
        pool_dim: str = "freq",
    ) -> None:
        super().__init__()
        self.n_filt = n_filt
        self.n_filt_last = n_filt[-1]
        cnn = torch.nn.Sequential()

        def conv(i: int, normalization: str = "batch", dropout: float | None = None, activ: str = "relu") -> None:
            in_dim = n_input_ch if i == 0 else n_filt[i - 1]
            out_dim = n_filt[i]
            if DY_layers[i] == 1:
                cnn.add_module(
                    f"conv{i}",
                    Dynamic_conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i], n_basis_kernels=n_basis_kernels, temperature=temperature, pool_dim=pool_dim),
                )
            else:
                cnn.add_module(f"conv{i}", torch.nn.Conv2d(in_dim, out_dim, kernel[i], stride[i], pad[i]))
            if normalization == "batch":
                cnn.add_module(f"batchnorm{i}", torch.nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.99))
            elif normalization == "layer":
                cnn.add_module(f"layernorm{i}", torch.nn.GroupNorm(1, out_dim))

            if activ.lower() == "leakyrelu":
                cnn.add_module(f"Relu{i}", torch.nn.LeakyReLu(0.2))
            elif activ.lower() == "relu":
                cnn.add_module(f"Relu{i}", torch.nn.ReLu())
            elif activ.lower() == "glu":
                cnn.add_module(f"glu{i}", GLU(out_dim))
            elif activ.lower() == "cg":
                cnn.add_module(f"cg{i}", ContextGating(out_dim))

            if dropout is not None:
                cnn.add_module(f"dropout{i}", torch.nn.Dropout(dropout))

        for i in range(len(n_filt)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            cnn.add_module(f"pooling{i}", torch.nn.AvgPool2d(pooling[i]))
        self.cnn = cnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x size : [bs, chan, frames, freqs]
        return self.cnn(x)


class BiGRU(torch.nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1) -> None:
        super().__init__()
        self.rnn = torch.nn.GRU(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)

    def forward(self, x):
        # self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        return x


class CRNN(torch.nn.Module):
    def __init__(self, n_input_ch, n_class=10, activation="glu", conv_dropout=0.5, n_RNN_cell=128, n_RNN_layer=2, rec_dropout=0, attention=True, **convkwargs) -> None:
        super().__init__()
        self.n_input_ch = n_input_ch
        self.attention = attention
        self.n_class = n_class

        self.cnn = CNN(n_input_ch=n_input_ch, activation=activation, conv_dropout=conv_dropout, **convkwargs)
        self.rnn = BiGRU(n_in=self.cnn.n_filt[-1], n_hidden=n_RNN_cell, dropout=rec_dropout, num_layers=n_RNN_layer)

        self.dropout = torch.nn.Dropout(conv_dropout)
        self.sigmoid = torch.nn.Sigmoid()
        self.dense = torch.nn.Linear(n_RNN_cell * 2, 10)

        if self.attention:
            self.dense_softmax = torch.nn.Linear(n_RNN_cell * 2, 10)
            if self.attention == "time":
                self.softmax = torch.nn.Softmax(dim=1)  # softmax on time dimension
            elif self.attention == "class":
                self.softmax = torch.nn.Softmax(dim=-1)  # softmax on class dimension
        # Load the pre-trained weights
        if torch.cuda.is_available():
            pretrained = torch.load("/home/tolga/Downloads/best_student.pt")
            self.load_state_dict(pretrained)
        # Now update the last layers to use the proper output classes
        self.dense = torch.nn.Linear(n_RNN_cell * 2, n_class)
        if self.attention:
            self.dense_softmax = torch.nn.Linear(n_RNN_cell * 2, n_class)

    def forward(self, x):  # input size : [bs, freqs, frames]
        if len(x.shape) == 4:
            x = x.squeeze(1)
        # cnn
        if self.n_input_ch > 1:
            x = x.transpose(2, 3)
        else:
            x = x.transpose(1, 2).unsqueeze(1)  # x size : [bs, chan, frames, freqs]
        x = self.cnn(x)
        bs, ch, frame, freq = x.size()
        if freq != 1:
            print("warning! frequency axis is large: " + str(freq))
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frame, ch * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # x size : [bs, frames, chan]

        # rnn
        x = self.rnn(x)  # x size : [bs, frames, 2 * chan]
        x = self.dropout(x)

        # classifier
        strong = self.dense(x)  # strong size : [bs, frames, n_class]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x)  # sof size : [bs, frames, n_class]
            sof = self.softmax(sof)  # sof size : [bs, frames, n_class]
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, n_class]
        else:
            weak = strong.mean(1)

        return weak  # strong.transpose(1, 2), weak
