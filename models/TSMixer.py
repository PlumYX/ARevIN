import torch
import torch.nn as nn

from layers.Invertible import RevIN, AdaRevIN

class Mlp_feat(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., activation='gelu'):
        super(Mlp_feat, self).__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU() if activation == "relu" else nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Mlp_time(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., activation='gelu'):
        super(Mlp_time, self).__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU() if activation == "relu" else nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.act(self.fc1(x)))


class Mixer_Layer(nn.Module):
    def __init__(self, time_dim, feat_dim, drop=0., activation='gelu', fac_C=False):
        super(Mixer_Layer, self).__init__()

        # nn.BatchNorm: axis is Integer
        # nn.LayerNorm: axis is Integer or List[Integer]
        # given [B, L, D] nn.BatchNorm1d(B) is equal to nn.LayerNorm([L, D])

        # self.batchNorm2D = nn.LayerNorm([time_dim, feat_dim]) # the norm of the paper, seems bad
        self.batchNorm2D = nn.BatchNorm1d(time_dim)
        self.MLP_time = Mlp_time(time_dim, time_dim, drop=drop, activation=None)
        self.MLP_feat = Mlp_feat(feat_dim, feat_dim, drop=drop, activation=None) if fac_C else None

    def forward(self, x):
        res1 = x
        x = self.batchNorm2D(x)
        x = self.MLP_time(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + res1

        if self.MLP_feat:
            res2 = x
            x = self.batchNorm2D(x)
            x = self.MLP_feat(x)
            x = x + res2
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        mix_layers = [Mixer_Layer(
            configs.seq_len, 
            configs.enc_in, 
            configs.dropout, 
            configs.activation, 
            configs.fac_C
            ) for _ in range(configs.e_layers)
        ]
        self.mix_layer = nn.Sequential(*mix_layers)
        self.temp_proj = nn.Linear(configs.seq_len, configs.pred_len)
        self.rev = RevIN(configs.enc_in) if configs.rev else None
        self.arev = AdaRevIN(configs.enc_in, mod=configs.arev_mode) if configs.arev else None

    def forecast(self, x):
        # x: [Batch, Input length, Channel]
        x = self.rev(x, 'norm') if self.rev else x
        x = self.arev(x, 'norm') if self.arev else x
        x = self.mix_layer(x)
        x = self.temp_proj(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.rev(x, 'denorm') if self.rev else x
        x = self.arev(x, 'denorm') if self.arev else x
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out  # [Batch, Output length, Channel]
