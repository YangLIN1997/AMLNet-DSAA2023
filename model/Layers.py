''' Define the Layers '''
import torch.nn as nn
import torch
from model.SubLayers import MultiHeadAttention, PositionwiseFeedForward
import torch.nn.functional as F



class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):        
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


class SSMLayer(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, L=20,S=20, dropout=0.1, device='cpu'):
        super().__init__()
        self.w_1 = nn.Linear(d_in, 1) # position-wise
        self.w_2 = nn.Linear(d_hid, 1) # position-wise        
        # self.w_o1 = nn.Linear(d_in, d_hid) # position-wise
        # self.w_o2 = nn.Linear(d_hid, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.S = S

        self.w_a_1 = nn.Linear(L*d_in, S) # position-wise
        self.w_a_2 = nn.Linear(d_hid, S) # position-wise
        self.layer_norm2 = nn.LayerNorm(L*d_in, eps=1e-6)

        self.w_ad_1 = nn.Linear(d_in, S) # position-wise
        self.w_ad_2 = nn.Linear(d_hid, S) # position-wise
        self.layer_normd2 = nn.LayerNorm(d_in, eps=1e-6)

        self.w_c_1 = nn.Linear(d_in, S) # position-wise
        self.w_c_2 = nn.Linear(d_hid, S) # position-wise
        self.layer_c = nn.LayerNorm(d_in, eps=1e-6)

        # self.w_d_1 = nn.Linear(d_in+2, 1) # position-wise
        # self.w_d_2 = nn.Linear(d_hid, 1) # position-wise
        # self.layer_d = nn.LayerNorm(d_in+2, eps=1e-6)
        self.distribution_sigma = nn.Softplus()

        # self.w_prelu_c_1 = torch.zeros(1,device=device)
        # self.w_prelu_a_1 = torch.zeros(1,device=device)
        # self.w_prelu_ad_1 = torch.zeros(1,device=device)

        self.device = device
        # self.w_w_cal_1 = nn.Linear(d_in+2, d_hid) # position-wise
        # self.w_w_cal_2 = nn.Linear(d_hid, 1) # position-wise
        # self.layer_w_cal = nn.LayerNorm(d_in+2, eps=1e-6)
        #
        # self.w_b_cal_1 = nn.Linear(d_in + 2, d_hid)  # position-wise
        # self.w_b_cal_2 = nn.Linear(d_hid, 5)  # position-wise
        # self.layer_b_cal = nn.LayerNorm(d_in + 2, eps=1e-6)

    def forward(self, x, alpha_0=None):
        # print('y',y.shape)
        n_batch = x.shape[0]
        L = x.shape[1]
        y_hat = torch.zeros(n_batch,L,1,device=self.device)
        alpha = torch.zeros(n_batch,L+1,self.S,1,device=self.device)
        Gamma = torch.zeros(n_batch,self.S,self.S,device=self.device)
        Gamma[:,2:,1:-1]=torch.eye(self.S-2,device=self.device)
        Gamma[:,1,1:] = -1
        Gamma[:,0,0] = 1

        z = torch.zeros(n_batch,self.S,1,device=self.device)
        z[:,0] = 1
        z[:,1] = 1
        Sigma = self.layer_norm(x)
        # Sigma = F.sigmoid(self.w_2(F.sigmoid(self.w_1(Sigma))))
        Sigma = self.distribution_sigma((self.w_1(Sigma)))
        # Sigma = self.dropout(Sigma)
        # return self.w_o2((self.w_o1(x))),alpha,Sigma

        c = self.layer_c(x)
        # c_temp = -0.5+F.hardsigmoid(self.w_c_2(-0.5+F.hardsigmoid(self.w_c_1(c_temp))))
        c = -0.5+F.hardsigmoid(self.w_c_1(c))
        # c = F.hardtanh(self.w_c_1(c))
        # c_temp = (self.w_c_2(F.relu(self.w_c_1(c_temp))))
        # c_temp = self.dropout(c_temp)

        if alpha_0 == None:
            a_0 = self.layer_norm2(x.view(n_batch,-1))
            # a_0 = self.w_a_2(F.relu(self.w_a_1(a_0)))
            # a_0 = -0.5+F.hardsigmoid(self.w_a_2(-0.5+F.hardsigmoid(self.w_a_1(a_0))))
            a_0 = -0.5+F.hardsigmoid(self.w_a_1(a_0))
            # a_0 = F.hardtanh(self.w_a_1(a_0))
            # a_0 = F.hardsigmoid(self.w_a_2(F.hardsigmoid(self.w_a_1(a_0))))
            # a_0 = (self.w_a_2(F.relu(self.w_a_1(a_0))))
            # a_0 = self.dropout(a_0)
            alpha[:,0] = a_0.unsqueeze_(-1)
        else:
            # alpha[:,0] = alpha_0[:,-1]
            a_0 = self.layer_normd2(x[:,0,:])
            # a_0 = self.w_a_2(F.relu(self.w_a_1(a_0)))
            # a_0 = -0.5+F.hardsigmoid(self.w_ad_2(-0.5+F.hardsigmoid(self.w_ad_1(a_0))))
            a_0 = -0.5+F.hardsigmoid(self.w_ad_1(a_0))
            # a_0 = F.hardtanh(self.w_ad_1(a_0))
            # a_0 = F.hardsigmoid(self.w_ad_2(F.hardsigmoid(self.w_ad_1(a_0))))
            # a_0 = (self.w_ad_2(F.relu(self.w_ad_1(a_0))))
            # a_0 = self.dropout(a_0)
            alpha[:,0] = a_0.unsqueeze_(-1)#+ alpha_0[:,-1]

        for i in range(L):
            alpha[:,i+1] = torch.matmul(Gamma,alpha[:,i] ) + c[:,i].unsqueeze_(-1)
            # y_hat[:,i] = torch.matmul(z.transpose(-1,-2),alpha[:,i+1]).squeeze_(-1) #+ d[:,i] #+ I[:,i].unsqueeze_(-1)
            y_hat[:,i] = alpha[:,i+1,0] + alpha[:,i+1,1]
            # y_hat[:,i] = torch.matmul(z.transpose(-1,-2),alpha[:,i]).squeeze_(-1)#+sigma[:,i]
            # alpha[:,i+1] = torch.matmul(Gamma,alpha[:,i]) + c[:,i].unsqueeze_(-1) + torch.matmul(R,eps[:,i].unsqueeze_(-1))

        # cal_in = torch.cat((x,torch.cat((y_hat,Sigma),-1)),-1)
        # w_cal = self.layer_w_cal(cal_in)
        # w_cal = (self.w_w_cal_2(F.relu(self.w_w_cal_1(w_cal))))
        #
        # b_cal = self.layer_b_cal(cal_in)
        # b_cal = (self.w_b_cal_2(F.relu(self.w_b_cal_1(b_cal))))
        #
        # cal = torch.cat((w_cal,b_cal),-1)
        # print(x.shape,alpha.shape,alpha[:,1:,:2].unsqueeze_(-1).shape)

        # d_temp = self.layer_d(torch.cat((x,alpha[:,1:,:2].squeeze_(-1)),axis=-1))
        # y_hat = y_hat+-0.5+F.sigmoid(self.w_d_2(-0.5+F.sigmoid(self.w_d_1(d_temp))))

        return y_hat,alpha,Sigma

# ''' Define the Layers '''
# import torch.nn as nn
# import torch
# from model.SubLayers import MultiHeadAttention, PositionwiseFeedForward
# import torch.nn.functional as F


# __author__ = "Yu-Hsiang Huang"


# class EncoderLayer(nn.Module):
#     ''' Compose with two layers '''

#     def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
#         super(EncoderLayer, self).__init__()
#         self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
#         self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

#     def forward(self, enc_input, slf_attn_mask=None):
#         enc_output, enc_slf_attn = self.slf_attn(
#             enc_input, enc_input, enc_input, mask=slf_attn_mask)
#         enc_output = self.pos_ffn(enc_output)
#         return enc_output, enc_slf_attn


# class DecoderLayer(nn.Module):
#     ''' Compose with three layers '''

#     def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
#         super(DecoderLayer, self).__init__()
#         self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
#         self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
#         self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

#     def forward(
#             self, dec_input, enc_output,
#             slf_attn_mask=None, dec_enc_attn_mask=None):
#         dec_output, dec_slf_attn = self.slf_attn(
#             dec_input, dec_input, dec_input, mask=slf_attn_mask)
#         dec_output, dec_enc_attn = self.enc_attn(
#             dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
#         dec_output = self.pos_ffn(dec_output)
#         return dec_output, dec_slf_attn, dec_enc_attn


# class SSMLayer(nn.Module):
#     ''' A two-feed-forward-layer module '''

#     def __init__(self, d_in, d_hid, L=20,S=20, dropout=0.1, device='cpu'):
#         super().__init__()
#         self.w_1 = nn.Linear(d_in, d_hid) # position-wise
#         self.w_2 = nn.Linear(d_hid, 1) # position-wise
#         self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
#         self.dropout = nn.Dropout(dropout)
#         self.S = S

#         self.w_a_1 = nn.Linear(L*d_in, d_hid) # position-wise
#         self.w_a_2 = nn.Linear(d_hid, S) # position-wise
#         self.layer_norm2 = nn.LayerNorm(L*d_in, eps=1e-6)

#         self.w_ad_1 = nn.Linear(d_in, d_hid) # position-wise
#         self.w_ad_2 = nn.Linear(d_hid, S) # position-wise
#         self.layer_normd2 = nn.LayerNorm(d_in, eps=1e-6)

#         self.w_c_1 = nn.Linear(d_in, d_hid) # position-wise
#         self.w_c_2 = nn.Linear(d_hid, S) # position-wise
#         self.layer_c = nn.LayerNorm(d_in, eps=1e-6)

#         self.w_d_1 = nn.Linear(d_in+2, d_hid) # position-wise
#         self.w_d_2 = nn.Linear(d_hid, 1) # position-wise
#         self.layer_d = nn.LayerNorm(d_in+2, eps=1e-6)
#         self.distribution_sigma = nn.Softplus()

#         self.device = device
#         # self.w_w_cal_1 = nn.Linear(d_in+2, d_hid) # position-wise
#         # self.w_w_cal_2 = nn.Linear(d_hid, 1) # position-wise
#         # self.layer_w_cal = nn.LayerNorm(d_in+2, eps=1e-6)
#         #
#         # self.w_b_cal_1 = nn.Linear(d_in + 2, d_hid)  # position-wise
#         # self.w_b_cal_2 = nn.Linear(d_hid, 5)  # position-wise
#         # self.layer_b_cal = nn.LayerNorm(d_in + 2, eps=1e-6)

#     def forward(self, x, alpha_0=None):
#         # print('y',y.shape)
#         n_batch = x.shape[0]
#         L = x.shape[1]
#         y_hat = torch.zeros(n_batch,L,1,device=self.device)
#         alpha = torch.zeros(n_batch,L+1,self.S,1,device=self.device)
#         Gamma = torch.zeros(n_batch,self.S,self.S,device=self.device)
#         Gamma[:,2:,1:-1]=torch.eye(self.S-2,device=self.device)
#         Gamma[:,1,1:] = -1
#         Gamma[:,0,0] = 1

#         z = torch.zeros(1,self.S,1,device=self.device)
#         z[:,0] = 1
#         z[:,1] = 1
#         Sigma = self.layer_norm(x)
#         # Sigma = F.sigmoid(self.w_2(F.sigmoid(self.w_1(Sigma))))
#         Sigma = self.distribution_sigma(self.w_2(F.sigmoid(self.w_1(Sigma))))
#         Sigma = self.dropout(Sigma)

#         c_temp = self.layer_c(x)
#         # c_temp = -0.5+F.sigmoid(self.w_c_2(-0.5+F.sigmoid(self.w_c_1(c_temp))))
#         c_temp = F.elu(self.w_c_2(F.relu(self.w_c_1(c_temp))))
#         c_temp = self.dropout(c_temp)
#         c=c_temp

#         if alpha_0 == None:
#             a_0 = self.layer_norm2(x.view(n_batch,-1))
#             # a_0 = self.w_a_2(F.relu(self.w_a_1(a_0)))
#             # a_0 = -0.5+F.sigmoid(self.w_a_2(-0.5+F.sigmoid(self.w_a_1(a_0))))
#             a_0 = F.elu(self.w_a_2(F.relu(self.w_a_1(a_0))))
#             a_0 = self.dropout(a_0)
#             alpha[:,0] = a_0.unsqueeze_(-1)
#         else:
#             # alpha[:,0] = alpha_0[:,-1]
#             a_0 = self.layer_normd2(x[:,0,:])
#             # a_0 = self.w_a_2(F.relu(self.w_a_1(a_0)))
#             # a_0 = -0.5+F.sigmoid(self.w_ad_2(-0.5+F.sigmoid(self.w_ad_1(a_0))))
#             a_0 = F.elu(self.w_ad_2(F.relu(self.w_ad_1(a_0))))
#             a_0 = self.dropout(a_0)
#             alpha[:,0] = a_0.unsqueeze_(-1)+ alpha_0[:,-1]

#         for i in range(L):
#             alpha[:,i+1] = torch.matmul(Gamma,alpha[:,i] ) + c[:,i].unsqueeze_(-1)
#             y_hat[:,i] = torch.matmul(z.transpose(-1,-2),alpha[:,i+1]).squeeze_(-1) #+ d[:,i] #+ I[:,i].unsqueeze_(-1)
#             # y_hat[:,i] = torch.matmul(z.transpose(-1,-2),alpha[:,i]).squeeze_(-1)#+sigma[:,i]
#             # alpha[:,i+1] = torch.matmul(Gamma,alpha[:,i]) + c[:,i].unsqueeze_(-1) + torch.matmul(R,eps[:,i].unsqueeze_(-1))

#         # cal_in = torch.cat((x,torch.cat((y_hat,Sigma),-1)),-1)
#         # w_cal = self.layer_w_cal(cal_in)
#         # w_cal = (self.w_w_cal_2(F.relu(self.w_w_cal_1(w_cal))))
#         #
#         # b_cal = self.layer_b_cal(cal_in)
#         # b_cal = (self.w_b_cal_2(F.relu(self.w_b_cal_1(b_cal))))
#         #
#         # cal = torch.cat((w_cal,b_cal),-1)
#         # print(x.shape,alpha.shape,alpha[:,1:,:2].unsqueeze_(-1).shape)

#         # d_temp = self.layer_d(torch.cat((x,alpha[:,1:,:2].squeeze_(-1)),axis=-1))
#         # y_hat = y_hat+-0.5+F.sigmoid(self.w_d_2(-0.5+F.sigmoid(self.w_d_1(d_temp))))

#         return y_hat,alpha,Sigma