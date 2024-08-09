import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from util.masking import TriangularCausalMask, ProbMask, PMask,PMask_test
from model.encoder import Encoder, EncoderLayer, ConvLayer
from model.decoder import Decoder, DecoderLayer
from model.attn import FullAttention, ProbAttention, AttentionLayer
from model.embed import DataEmbedding,DataEmbedding_AR
import torchsort

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s, _ = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, n_g=5,alpha_L=0.05,
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                dropout=0.0, attn='prob', embed='fixed', data='ETTh', activation='gelu', 
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        embed = 'fixed' if embed == 0 else 'nnEmbedding'
        attn = 'prob' if attn == 0 else 'Full'
        activation = 'gelu' if activation == 0 else 'relu'
        self.label_len = label_len
        self.pred_len = out_len
        self.attn = attn
        self.n_g = n_g
        self.alpha_L=alpha_L
        self.device=device
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, data, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, data, dropout)
        self.dec_embedding_T = DataEmbedding(dec_in, d_model, embed, data, dropout)
        self.dec_embedding_S = DataEmbedding(dec_in-1, d_model, embed, data, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder_DAR = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout),
                                d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout),
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection_DAR_mu = nn.Linear(d_model, c_out, bias=True)
        self.projection_DAR_presigma = nn.Linear(d_model, c_out, bias=True)
        self.projection_DAR_sigma = nn.Softplus()

        # Decoder T
        self.decoder_T = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout),
                                d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout),
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection_T = nn.Linear(d_model, c_out, bias=True)

        # Decoder S
        self.decoder_S = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=dropout),
                                d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout),
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(1)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection_S = nn.Linear(d_model, c_out, bias=True)


    def forward_DAR(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)


        T=int(self.pred_len/self.n_g)

        # dec_T
        dec_out_T = self.dec_embedding_T(x_dec, x_mark_dec)
        dec_out_T,hidden_T = self.decoder_T(dec_out_T, enc_out, x_mask=None, cross_mask=dec_enc_mask)
        interd_T = self.projection_T(dec_out_T[:,-self.pred_len+1:,:]).squeeze(-1)
        interd_T=torch.nn.functional.softmax(interd_T)
        indices_T = torchsort.soft_rank(interd_T, regularization_strength=0.0001).int() - 1

        indices_T = indices_T + self.label_len + 1
        indices_T = torch.cat((torch.zeros((indices_T.shape[0], 1), dtype=torch.long, device=interd_T.device) + (self.label_len), indices_T),dim=-1)
        # dec_S
        dec_out_S = self.dec_embedding_S(x_dec[:,:,1:], x_mark_dec)
        dec_out_S,hidden_S = self.decoder_S(dec_out_S, enc_out, x_mask=None, cross_mask=dec_enc_mask)
        interd_S = self.projection_S(dec_out_S[:,-self.pred_len+1:,:]).squeeze(-1)
        indices_S = torchsort.soft_rank(interd_S)[:, :self.n_g] - 1
        indices_S = indices_S + self.label_len + 1
        indices_S = torch.cat((torch.zeros((indices_S.shape[0],1),dtype=torch.long,device=interd_S.device)+(self.label_len),indices_S), dim=-1)

        # dec
        dec_out_DAR = self.dec_embedding(x_dec, x_mark_dec)

        dec_out_DAR,hidden_DAR = self.decoder_DAR(dec_out_DAR, enc_out, x_mask=dec_self_mask_DAR, cross_mask=dec_enc_mask)
        mu_DAR = self.projection_DAR_mu(dec_out_DAR[:,-self.pred_len:,:]).squeeze(-1)
        pre_sigma = self.projection_DAR_presigma(dec_out_DAR[:,-self.pred_len:,:])
        sigma_DAR = self.projection_DAR_sigma(pre_sigma).squeeze(-1)

        return mu_DAR,sigma_DAR,hidden_DAR,indices_S,T,indices_T  # [B, L, D]


    def test_DAR(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)


        label=torch.zeros((1,self.pred_len-1),device=enc_out.device)+self.alpha_L
        T=int(self.pred_len/self.n_g)
        for i in range(self.n_g-1):
            label[:,-1+T*(i+1)]=1-self.alpha_L
        # dec_S
        dec_out_S = self.dec_embedding_S(x_dec[:,:,1:], x_mark_dec)
        dec_out_S,hidden_S = self.decoder_S(dec_out_S, enc_out, x_mask=None, cross_mask=dec_enc_mask)
        interd_S = self.projection_S(dec_out_S[:,-self.pred_len+1:,:]).squeeze(-1)
        interd_S = interd_S * label
        indices_S = torchsort.soft_rank(interd_S)[:, :self.n_g].int() - 1
        indices_S = indices_S + self.label_len + 1
        up_bound=torch.cat((indices_S,torch.zeros((indices_S.shape[0],1),dtype=torch.long,device=interd_S.device)+(self.label_len+self.pred_len)), dim=-1)
        indices_S = torch.cat((torch.zeros((indices_S.shape[0],1),dtype=torch.long,device=interd_S.device)+(self.label_len),indices_S), dim=-1)
        steps=torch.max(up_bound-indices_S)

        #  dec
        dec_self_mask_DAR = PMask_test(x_dec.shape[0], self.label_len, self.pred_len, indices_S,device=interd_S.device)
        for t in range(steps):
            dec_out_DAR = self.dec_embedding(x_dec, x_mark_dec)
            dec_self_mask_DAR.forward()
            dec_out_DAR,hidden_DAR = self.decoder_DAR(dec_out_DAR, enc_out, x_mask=dec_self_mask_DAR, cross_mask=dec_enc_mask)
            mu_DAR = self.projection_DAR_mu(dec_out_DAR).squeeze(-1)
            x_dec[:,-self.pred_len:,0] = mu_DAR[:,-self.pred_len-1:-1]

        pre_sigma = self.projection_DAR_presigma(dec_out_DAR)
        sigma_DAR = self.projection_DAR_sigma(pre_sigma).squeeze(-1)
        return mu_DAR[:,self.label_len:],sigma_DAR[:,self.label_len:],hidden_DAR,indices_S  # [B, L, D]


##############################
       # Loss
##############################

def loss_fn(mu, sigma, labels, predict_start):
    labels = labels[:,predict_start:]
    mask = sigma == 0
    sigma_index = ~mask
    distribution = torch.distributions.normal.Normal(mu[sigma_index], sigma[sigma_index])
    likelihood = distribution.log_prob(labels[sigma_index])
    return -torch.mean(likelihood)

def loss_fn_L(indices):
    mean = torch.mean(torch.Tensor.float(indices),dim=-1).reshape(-1,1)
    var=torch.mean((torch.Tensor.float(indices)-mean)**2/mean)
    return var
def loss_fn_T(indices,indices_T):
    return torch.mean((indices-indices_T).float()**2)

