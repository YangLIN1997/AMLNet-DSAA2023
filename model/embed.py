import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # print(x.dtype)
        # print(self.tokenConv.weight.dtype)
        # x = x.type(self.tokenConv.weight.dtype)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TokenEmbedding_AR(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding_AR, self).__init__()
        # padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=2)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # print(x.shape)
        # print(x.dtype)
        # print(self.tokenConv.weight.dtype)
        # x = x.type(self.tokenConv.weight.dtype)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        # print(x.shape)

        return x[:,:-2,:]


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', data='ETTh'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 60;
        hour_size = 24
        weekday_size = 7;
        day_size = 32;
        month_size = 13
        self.data = data
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if self.data == 'Sanyo' or self.data == 'Hanergy':
            self.minute_embed = Embed(minute_size, d_model)
            self.hour_embed = Embed(hour_size, d_model)
            self.month_embed = Embed(month_size, d_model)
        elif self.data == 'Solar':
            self.hour_embed = Embed(hour_size, d_model)
            self.month_embed = Embed(month_size, d_model)
            self.age_embed = Embed(5833, d_model)
            self.id_embed = Embed(137, d_model)
        elif self.data == 'traffic':
            self.hour_embed = Embed(hour_size, d_model)
            self.weekday_embed = Embed(weekday_size, d_model)
            self.month_embed = Embed(month_size, d_model)
            self.age_embed = Embed(10921, d_model)
            self.id_embed = Embed(963, d_model)
        elif self.data == 'exchange_rate':
            self.weekday_embed = Embed(5, d_model)
            self.month_embed = Embed(month_size, d_model)
            self.age_embed = Embed(7045, d_model)
            self.id_embed = Embed(8, d_model)
        else: #elect
            self.weekday_embed = Embed(weekday_size, d_model)
            self.hour_embed = Embed(hour_size, d_model)
            self.month_embed = Embed(month_size, d_model)
            self.age_embed = Embed(32305, d_model)
            self.id_embed = Embed(370, d_model)

    def forward(self, x):
        x = x.long()
        # print(torch.max(x[:, :, -3]),torch.min(x[:, :, -3]))
        if self.data == 'Sanyo' or self.data == 'Hanergy':
            minute_x = self.minute_embed(x[:, :, -3]) if hasattr(self, 'minute_embed') else 0.
            hour_x = self.hour_embed(x[:, :, -2])
            month_x = self.month_embed(x[:, :, -1])
            return hour_x + month_x + minute_x
        elif self.data == 'Solar':
            hour_x = self.hour_embed(x[:, :, 0])
            month_x = self.month_embed(x[:, :, 1])
            age_x = self.age_embed(x[:, :, 2])
            id_x = self.id_embed(x[:, :, 3])
            return hour_x + month_x + age_x + id_x
        elif self.data == 'Traffic':
            hour_x = self.hour_embed(x[:, :, 1])
            weekday_x = self.weekday_embed(x[:, :, 0])
            month_x = self.month_embed(x[:, :, 2])
            age_x = self.age_embed(x[:, :, 3])
            id_x = self.id_embed(x[:, :, 4])
            return hour_x + weekday_x + month_x + age_x + id_x
        elif self.data == 'exchange_rate':
            weekday_x = self.weekday_embed(x[:, :, 0])
            month_x = self.month_embed(x[:, :, 1])
            age_x = self.age_embed(x[:, :, 2])
            id_x = self.id_embed(x[:, :, 3])
            return weekday_x + month_x + age_x + id_x
        else:
            weekday_x = self.weekday_embed(x[:, :, 0])
            hour_x = self.hour_embed(x[:, :, 1])
            month_x = self.month_embed(x[:, :, 2])
            age_x = self.age_embed(x[:, :, 3])
            id_x = self.id_embed(x[:, :, 4])
            return weekday_x + hour_x + month_x + age_x + id_x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', data='ETTh', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, data=data)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # print(x.shape)
        # print(x_mark.shape)
        # print(x[0])
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

class DataEmbedding_AR(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', data='ETTh', dropout=0.1):
        super(DataEmbedding_AR, self).__init__()
        self.value_embedding = TokenEmbedding_AR(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, data=data)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # print(x.shape)
        # print(x_mark.shape)
        # print(x[0])
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)