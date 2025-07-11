import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class AdaRevIN(nn.Module):
    def __init__(self, num_features: int, mod='type0', arev_param=[0.5, 0.5], eps=1e-5, affine=True, arev_affine=True):
        super(AdaRevIN, self).__init__()

        self.num_features = num_features
        self.mod = mod
        self.arev_param = arev_param
        self.eps = eps
        self.affine = affine
        self.arev_affine = arev_affine
        self.channel_individual = False
        if self.affine:
            self._init_params()
        self._init_arev_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            if self.mod == 'type0':
                x = self.arev_weight[0] * x + self._denormalize(self.arev_weight[1] * x)
            elif self.mod == 'type1':
                x = self.arev_weight[0] * x + self.arev_weight[1] * self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _init_arev_params(self):
        self.arev_weight = nn.Parameter(
            torch.Tensor(2, self.num_features)
            ) if self.channel_individual else nn.Parameter(torch.Tensor(2))
        torch.nn.init.constant_(self.arev_weight[0], self.arev_param[0])
        torch.nn.init.constant_(self.arev_weight[1], self.arev_param[1])
        if self.arev_affine == False:
           self.arev_weight.requires_grad_(False)

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
