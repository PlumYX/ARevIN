import torch
import torch.nn as nn

from layers.Invertible import RevIN, AdaRevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.pred_len = configs.pred_len

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/configs.seq_len)*torch.ones([configs.pred_len,configs.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(configs.seq_len, configs.pred_len))
        else:
            self.Linear = nn.Linear(configs.seq_len, configs.pred_len)
        self.rev = RevIN(configs.enc_in) if configs.rev else None
        self.arev = AdaRevIN(configs.enc_in, mod=configs.arev_mode) if configs.arev else None

    def forecast(self, x):
        # x: [Batch, Input length, Channel]
        x = self.rev(x, 'norm') if self.rev else x
        x = self.arev(x, 'norm') if self.arev else x
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.rev(x, 'denorm') if self.rev else x
        x = self.arev(x, 'denorm') if self.arev else x
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out  # [Batch, Output length, Channel]
