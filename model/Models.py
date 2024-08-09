''' Define the Transformer model '''
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from model.Layers import EncoderLayer, DecoderLayer
import torch.nn.functional as F
import math


def get_pad_mask(seq, pad_idx):
    # print(seq.shape)
    return (seq != pad_idx)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s, _ = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=50):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x, device):
        position = torch.HalfTensor(np.zeros((x.shape[0],x.shape[1],self.pos_table.shape[2]))).to(device)

        position = position + self.pos_table[:, :x.size(1)].clone().detach()
        return torch.cat( (x, position),dim=2)


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,id_enc, d_embedding, n_layers, n_head, d_k, d_v,
            d_model, d_inner, L=20, S=20, dropout=0.1, n_position=200, device='cpu'):

        super().__init__()

        # self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.d_embedding = d_embedding
        self.id_enc = id_enc
        self.position_enc = PositionalEncoding(d_embedding, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.device = device

    def forward(self, id, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []
        # -- Forward
        enc_output = self.position_enc(src_seq, self.device)        

        if id is not None:
            enc_output[:,:,-self.d_embedding:]+=self.id_enc(id)
        enc_output = self.dropout(enc_output)

        src_mask = None
        # print(enc_output.shape,src_mask.shape)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        enc_output = self.layer_norm(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output#, y_hat, alpha,Sigma


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,id_enc, d_embedding, n_layers, n_head, d_k, d_v,
            d_model, d_inner, n_position=200, L=20, S=20, dropout=0.1, device='cpu'):

        super().__init__()
        # self.trg_word_emb = nn.Embedding(115, 20, padding_idx=0)
        self.d_embedding = d_embedding
        self.id_enc = id_enc
        self.position_enc = PositionalEncoding(d_embedding, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.device = device

    def forward(self, id, trg_seq, trg_mask, enc_output, y_hat, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.position_enc(trg_seq, self.device)
        if id is not None:
            dec_output[:,:,-self.d_embedding:]+=self.id_enc(id)
        dec_output = self.dropout(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        dec_output = self.layer_norm(dec_output)
        # print(dec_enc_attn_list[-1].shape)
        if return_attns:
            return dec_output, y_hat, dec_slf_attn_list, dec_enc_attn_list
        return dec_output


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, params, dataset):
        # , n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
        # d_word_vec=8, d_model=12+8, d_inner=20,
        # n_layers=2, n_head=4, d_k=6, d_v=6, dropout=0.1, n_position=40,
        # trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True):

        super().__init__()

        # self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.params = params
        self.dataset = dataset
        self.id_enc = None
        if self.dataset != 'Sanyo' and self.dataset != 'Hanergy':
            self.id_enc = torch.nn.Embedding(params.n_id, params.d_embedding, padding_idx=0)

        self.encoder = Encoder(
            id_enc=self.id_enc,n_position=params.n_position,
            d_embedding=params.d_embedding, d_model=params.d_model, d_inner=params.d_inner,
            n_layers=params.n_layers, n_head=params.n_head, d_k=params.d_k, d_v=params.d_v,
            L=params.predict_start,S=params.S,
            dropout=params.dropout, device = params.device)

        self.decoder = Decoder(
            id_enc=self.id_enc,n_position=params.n_position,
            d_embedding=params.d_embedding, d_model=params.d_model, d_inner=params.d_inner,
            n_layers=params.n_layers, n_head=params.n_head, d_k=params.d_k, d_v=params.d_v,
            L=params.predict_steps, S=params.S,
            dropout=params.dropout, device = params.device)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # self.conv = torch.nn.Conv1d(in_channels=params.d_model,out_channels=params.d_model,kernel_size=2,stride=1,padding=0,bias=False)
        # self.conv.weight.data[:,0,0]=torch.ones((params.d_model))/2
        # self.conv.weight.data[:,0,1]=torch.ones((params.d_model))/2
        self.conv_t = []
        for i in range(self.params.n_cnn_layers):
            self.conv_t += [torch.nn.Conv1d(in_channels=1,out_channels=1,kernel_size=self.params.kernel_size,stride=self.params.stride,padding=0,bias=False)]
            self.conv_t[i].weight.data=torch.ones((self.conv_t[i].weight.data.shape))/self.params.kernel_size
            # self.conv_t[i].weight.requires_grad = False
        self.linear_t = torch.nn.Linear(self.params.predict_start-self.params.n_cnn_layers*self.params.stride*(self.params.kernel_size-1),1,bias=False)
        self.linear_t.weight.data = torch.ones((self.linear_t.weight.data.shape))

        self.linear_s_in1 = torch.nn.Linear(self.params.d_model,params.d_model//2,bias=True)
        self.linear_s_in2 = torch.nn.Linear(self.params.d_model//2,1,bias=True)
        self.conv_s = []
        for i in range(self.params.n_cnn_layers):
            self.conv_s += [torch.nn.Conv1d(in_channels=1,out_channels=1,kernel_size=self.params.kernel_size,stride=self.params.stride,padding=0,bias=False)]
            if i == 0:
                self.conv_s[i].weight.data[:, 0, 0] = -torch.ones((self.conv_s[i].weight.data[:, 0, 0].shape))
                self.conv_s[i].weight.data[:, 0, 1] = torch.zeros((self.conv_s[i].weight.data[:, 0, 1].shape))
            else:
                self.conv_s[i].weight.data = torch.ones((self.conv_s[i].weight.data.shape)) / self.params.kernel_size
        self.linear_s = torch.nn.Linear(self.params.predict_start-self.params.n_cnn_layers*self.params.stride*(self.params.kernel_size-1),1,bias=False)
        self.linear_s.weight.data = torch.ones((self.linear_t.weight.data.shape))

        self.conv_i = []
        for i in range(self.params.n_cnn_layers):
            self.conv_i += [torch.nn.Conv1d(in_channels=self.params.d_model,out_channels=self.params.d_model,kernel_size=self.params.kernel_size,stride=self.params.stride,padding=0,bias=True)]
            # self.conv_i[i].weight.data=torch.ones((self.conv_i[i].weight.data.shape))/self.params.kernel_size
        self.linear_i1 = torch.nn.Linear(self.params.d_model*(self.params.predict_start-self.params.n_cnn_layers*self.params.stride*(self.params.kernel_size-1)),1,bias=True)
        self.linear_i2 = torch.nn.Linear(self.params.d_model*(self.params.predict_start-self.params.n_cnn_layers*self.params.stride*(self.params.kernel_size-1)),1,bias=True)


    def forward(self, src_seq, trg_seq):

        src_mask = get_pad_mask(src_seq, -1)
        src_mask = None
        trg_mask = get_subsequent_mask(trg_seq)

        if (self.dataset == 'Sanyo') or (self.dataset == 'Hanergy'):
            enc_id = None
            dec_id = None
        else:
            enc_id = src_seq[:,:,-1].type(torch.long)
            dec_id = trg_seq[:, :, -1].type(torch.long)
            src_seq = src_seq[:,:,:-1]
            trg_seq = trg_seq[:,:,:-1]

        enc_output = self.encoder(enc_id, src_seq, src_mask,)
        dec_output = self.decoder(dec_id, trg_seq, trg_mask, enc_output, None, src_mask)

        srctrg_seq = torch.cat((src_seq,trg_seq),dim=1)
        encdec_output = torch.cat((enc_output,dec_output),dim=1)
        dec_Sigma = torch.zeros(trg_seq.shape[0],trg_seq.shape[1])
        dec_y_hat = torch.zeros(trg_seq.shape[0],trg_seq.shape[1])

        for t in range(self.params.predict_steps):
            in_conv_t = srctrg_seq[:,t+1:self.params.predict_start+1+t,0].unsqueeze(1)
            for i in range(self.params.n_cnn_layers):
                out_conv_t = self.conv_t[i](in_conv_t)
                in_conv_t = out_conv_t
            out_t = self.linear_t(out_conv_t)/self.linear_t.weight.data.shape[1]

            in_conv_s = torch.nn.functional.leaky_relu(self.linear_s_in1(encdec_output[:,t+1:self.params.predict_start+1+t]))
            in_conv_s = torch.nn.functional.leaky_relu(self.linear_s_in2(in_conv_s)).transpose(1,2)
            for i in range(self.params.n_cnn_layers):
                out_conv_s = self.conv_s[i](in_conv_s)
                in_conv_s = out_conv_s
            out_s = self.linear_t(out_conv_s)#/self.linear_s.weight.data.shape[1]
            # out_s=out_s-(torch.mean(out_s,axis=1).unsqueeze(-1))

            in_conv_i = encdec_output[:,t+1:self.params.predict_start+1+t].transpose(1,2)
            for i in range(self.params.n_cnn_layers):
                out_conv_i = self.conv_i[i](in_conv_i)
                in_conv_i = out_conv_i
            out_i1 = self.linear_i1(out_conv_i.reshape(out_conv_i.shape[0],-1))

            dec_Sigma[:,t] = torch.nn.functional.softplus(self.linear_i2(out_conv_i.reshape(out_conv_i.shape[0],-1))).squeeze(-1)
            dec_y_hat[:,t] = (out_t.squeeze(-1) + out_s.squeeze(-1) ).squeeze(-1)
        return dec_y_hat,dec_Sigma

    def test(self, src_seq, trg_seq):

        src_mask = get_pad_mask(src_seq, -1)
        src_mask = None
        trg_mask = get_subsequent_mask(trg_seq)

        if (self.dataset == 'Sanyo') or (self.dataset == 'Hanergy'):
            enc_id = None
            dec_id = None
        else:
            enc_id = src_seq[:,:,-1].type(torch.long)
            dec_id = trg_seq[:, :, -1].type(torch.long)
            src_seq = src_seq[:,:,:-1]
            trg_seq = trg_seq[:,:,:-1]
        # enc_output, enc_y_hat, enc_alpha, enc_Sigma= self.encoder(enc_id,src_seq, src_mask)
        enc_output = self.encoder(enc_id,src_seq, src_mask)

        srctrg_seq = torch.cat((src_seq,trg_seq),dim=1)
        dec_Sigma = torch.zeros(trg_seq.shape[0],trg_seq.shape[1])
        dec_y_hat = torch.zeros(trg_seq.shape[0],trg_seq.shape[1])
        dec_decom = torch.zeros(trg_seq.shape[0],trg_seq.shape[1],2)
        for t in range(self.params.predict_steps):
            if (self.dataset == 'Sanyo') or (self.dataset == 'Hanergy'):
                dec_output = self.decoder(dec_id,trg_seq[:,0:t+1,:], trg_mask[:,0:t+1,0:t+1], enc_output,
                                                        # enc_y_hat, enc_alpha, src_mask)
                                                        None, src_mask,False)
            else:
                # dec_output, dec_y_hat, dec_alpha, dec_Sigma = self.decoder(dec_id[:,0:t+1],trg_seq[:,0:t+1,:], trg_mask[:,0:t+1,0:t+1], enc_output,
                                                        # enc_y_hat, enc_alpha, src_mask)                
                dec_output = self.decoder(dec_id[:,0:t+1],trg_seq[:,0:t+1,:], trg_mask[:,0:t+1,0:t+1], enc_output,
                                                        None, src_mask,False)

            encdec_output = torch.cat((enc_output, dec_output), dim=1)
            in_conv_t = srctrg_seq[:,t+1:self.params.predict_start+1+t,0].unsqueeze(1)
            for i in range(self.params.n_cnn_layers):
                out_conv_t = self.conv_t[i](in_conv_t)
                in_conv_t = out_conv_t
            out_t = self.linear_t(out_conv_t)/self.linear_t.weight.data.shape[1]

            in_conv_s = torch.nn.functional.leaky_relu(self.linear_s_in1(encdec_output[:,t+1:self.params.predict_start+1+t]))
            in_conv_s = torch.nn.functional.leaky_relu(self.linear_s_in2(in_conv_s)).transpose(1,2)
            for i in range(self.params.n_cnn_layers):
                out_conv_s = self.conv_s[i](in_conv_s)
                in_conv_s = out_conv_s
            out_s = self.linear_t(out_conv_s)#/self.linear_s.weight.data.shape[1]
            # out_s=out_s-(torch.mean(out_s,axis=1).unsqueeze(-1))

            in_conv_i = encdec_output[:,t+1:self.params.predict_start+1+t].transpose(1,2)
            for i in range(self.params.n_cnn_layers):
                out_conv_i = self.conv_i[i](in_conv_i)
                in_conv_i = out_conv_i
            out_i1 = self.linear_i1(out_conv_i.reshape(out_conv_i.shape[0],-1))

            dec_Sigma[:,t] = torch.nn.functional.softplus(self.linear_i2(out_conv_i.reshape(out_conv_i.shape[0],-1))).squeeze(-1)
            dec_y_hat[:,t] = (out_t.squeeze(-1) + out_s.squeeze(-1) ).squeeze(-1)

            dec_decom[:,t,0] = out_t.squeeze(-1).squeeze(-1)
            dec_decom[:, t, 1] = out_s.squeeze(-1).squeeze(-1)

            if t < (self.params.predict_steps - 1):
                trg_seq[:,t+1,0] = dec_y_hat[:,-1]
        #       trg_seq[:,t+1,0] = seq_logit[:,-1,0]
        print(dec_decom[0,:,0])
        print(dec_decom[0,:,1])
        return dec_y_hat,dec_decom,dec_Sigma


# def loss_fn(mu: Variable,labels: Variable):
#     '''
#     Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
#     Args:
#         mu: (Variable) dimension [batch_size] - estimated mean at time step t
#         sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
#         labels: (Variable) dimension [batch_size] z_t
#     Returns:
#         loss: (Variable) average log-likelihood loss across the batch
#     '''
#     return torch.mean((mu - labels)**2)

def loss_fn(mu: Variable, labels: Variable, predict_start):
    labels = labels[:,predict_start:]
    zero_index = (labels != 0)
    # print(labels.shape,mu.shape)
    difference = (mu[zero_index] - labels[zero_index])**2
    return torch.mean(difference)


# if relative is set to True, metrics are not normalized by the scale of labels
def accuracy_ND(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    # print(labels.shape,mu.shape)
    if relative:
        diff = torch.mean(torch.abs(mu[zero_index] - labels[zero_index])).item()
        return [diff, 1]
    else:
        diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
        # diff = 2*torch.sum(torch.mul(0.9-(labels[zero_index]<mu[zero_index]).type(mu.dtype),
        #               labels[zero_index] - mu[zero_index])).item()
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        return[diff, summation]


def accuracy_RMSE(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    diff = torch.sum(torch.mul((mu[zero_index] - labels[zero_index]), (mu[zero_index] - labels[zero_index]))).item()
    if relative:
        return [diff, torch.sum(zero_index).item(), torch.sum(zero_index).item()]
    else:
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        if summation == 0:
            logger.error('summation denominator error! ')
        return [diff, summation, torch.sum(zero_index).item()]


def accuracy_ROU(mu: torch.Tensor, labels: torch.Tensor, rou: float,relative = False):
    zero_index = (labels != 0)
    if relative:
        diff = 2*torch.sum(torch.mul(rou-(labels[zero_index]<mu[zero_index]).type(mu.dtype),
                      labels[zero_index] - mu[zero_index])).item()
        return [diff, 1]
    else:
        # diff = torch.sum(torch.abs(mu[zero_index] - labels[zero_index])).item()
        diff = 2*torch.sum(torch.mul(rou-(labels[zero_index]<mu[zero_index]).type(mu.dtype),
                      labels[zero_index] - mu[zero_index])).item()
        summation = torch.sum(torch.abs(labels[zero_index])).item()
        return[diff, summation]

def accuracy_ROU(rou: float, mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels != 0)
    labels = labels[zero_index]
    mu = mu[zero_index]
    if relative:
        diff = 2*torch.mean(torch.mul(rou-(labels<mu).type(mu.dtype),
                      labels - mu)).item()
        return [diff, 1]
    else:
        diff = 2*torch.sum(torch.mul(rou-(labels<mu).type(mu.dtype),
                      labels - mu))
        summation = torch.sum(torch.abs(labels)).item()
        return[diff, summation]
    numerator = 0
    denominator = 0
    samples = mu
    pred_samples = mu.shape[0]
    for t in range(labels.shape[1]):
        zero_index = (labels[:, t] != 0)
        if zero_index.numel() > 0:
            rou_th = math.ceil(pred_samples * (1 - rou))
            rou_pred = torch.topk(samples[:, zero_index, t], dim=0, k=rou_th)[0][-1, :]
            abs_diff = labels[:, t][zero_index] - rou_pred
            numerator += 2 * (torch.sum(rou * abs_diff[labels[:, t][zero_index] > rou_pred]) - torch.sum(
                (1 - rou) * abs_diff[labels[:, t][zero_index] <= rou_pred])).item()
            denominator += torch.sum(labels[:, t][zero_index]).item()
    if relative:
        return [numerator, torch.sum(labels != 0).item()]
    else:
        return [numerator, denominator]

def accuracy_ROU_(rou: float, mu: torch.Tensor, labels: torch.Tensor, relative = False):
    zero_index = (labels == 0)
    mu[zero_index] = 0
    if relative:
        diff = 2*torch.mean(torch.mul(rou-(labels<mu).type(mu.dtype),
                      labels - mu), axis=1)
        return diff/1
    else:
        diff = 2*torch.sum(torch.mul(rou-(labels<mu).type(mu.dtype),
                      labels - mu), axis=1)
        summation = torch.sum(torch.abs(labels), axis=1)

        return diff/summation


def accuracy_ND_(mu: torch.Tensor, labels: torch.Tensor, relative = False):

    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mu[labels == 0] = 0.

    diff = np.sum(np.abs(mu - labels), axis=1)
    if relative:
        summation = np.sum((labels != np.inf), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result
    else:
        summation = np.sum(np.abs(labels), axis=1)
        mask = (summation == 0)
        summation[mask] = 1
        result = diff / summation
        result[mask] = -1
        return result


def accuracy_RMSE_(mu: torch.Tensor, labels: torch.Tensor, relative = False):
    mu = mu.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    mask = labels == 0
    mu[labels == 0] = 0.

    diff = np.sum((mu - labels) ** 2, axis=1)
    summation = np.sum(np.abs(labels), axis=1)
    mask2 = (summation == 0)
    if relative:
        div = np.sum(~mask, axis=1)
        div[mask2] = 1
        result = np.sqrt(diff / div)
        result[mask2] = -1
        return result
    else:
        summation[mask2] = 1
        result = (np.sqrt(diff) / summation) * np.sqrt(np.sum(~mask, axis=1))
        result[mask2] = -1
        return result
