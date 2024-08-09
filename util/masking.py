import torch

class PMask():
    def __init__(self, B, label_len,pred_len, indices,T, n_g ,device="cpu"):
        self._mask = torch.ones((B,1,1,label_len+pred_len), dtype=torch.bool,device=device)
        self._mask[:, 0, 0, :label_len] = 0
        self._mask = self._mask.repeat(1,1,label_len+pred_len,1)
        k=indices.shape[-1]

        # src = torch.zeros((B,label_len+pred_len), dtype=torch.bool,device=device)
        # # print(indices[0])
        # for i in range(label_len,label_len+pred_len):
        #     if i>label_len:
        #         a = indices[:,-k:]+1
        #         a[a>=label_len+pred_len]=label_len+pred_len-1
        #         indices=torch.cat((indices,a),dim=-1)
        #     for j in range(indices.shape[-1]):
        #         # print(indices[:,j].shape)
        #         # print(self._mask[:, 0, i].shape)
        #         self._mask[:, 0, i]=self._mask[:, 0, i].scatter_(-1,indices[:,j].reshape(-1,1),src)
            # print(indices[0])
            # print(self._mask[0, 0, i])
        # print(self._mask[0])

        for i in range(n_g):
            print(indices[0,:min((i+1)*n_g,pred_len)])
# print(indices[0, :min((i + 1) * n_g, pred_len)])
            temp = torch.ones((B,label_len+pred_len), dtype=torch.bool,device=device)
            temp[:,indices[:,:min((i+1)*n_g,pred_len)]]=0
            for j in range(T):
                # print('result', self._mask[:, :,indices[:, min(i * n_g + j, pred_len - 1)]].shape)
                # print( indices[:, min(i * n_g + j, pred_len - 1)].shape)
                # print('indices',indices[:, min(i * n_g + j, pred_len - 1)].shape)
                # print('_mask',self._mask.shape)
                self._mask[:, :, indices[:, min(i * n_g + j, pred_len - 1)]]=temp
                # self._mask[:, 0,indices[:,min(i*n_g+j,pred_len-1)]]=temp
                # self._mask[:, 0, indices[:, min(i * n_g + j, pred_len - 1)],indices[:,:min((i+1)*n_g,pred_len)]] = 0
                # self._mask[:, 0] = self._mask[:, 0].scatter_(-2, indices[:, j].reshape(-1, 1,1), temp)
            # self._mask[:, 0, indices[:, i * n_g : max(i * n_g + j, pred_len - 1)].reshape(-1, 1),indices[:, :max((i + 1) * n_g, pred_len)]] = 0
            #     print(self._mask[0, 0, indices[0,  min(i * n_g + j, pred_len - 1)].reshape(-1, 1)])
            #     print('indices', self._mask[:, :,indices[:, min(i * n_g + j, pred_len - 1)]].shape)
            #     print('result', self._mask[:, :,indices[:, min(i * n_g + j, pred_len - 1)]])

            # print(self._mask[:, 0, indices[:, i * n_g : max(i * n_g + j, pred_len - 1)].reshape(-1, 1),indices[:, :max((i + 1) * n_g, pred_len)]])
        # print(indices)
        print(self._mask[0])
        self._mask = self._mask.to(device)

    @property
    def mask(self):
        return self._mask

class PMask_test():
    def __init__(self, B, label_len,pred_len, indices,device="cpu"):

        self.mask_full = torch.ones((B,1,1,label_len+pred_len), dtype=torch.bool,device=device)
        self.mask_full[:, 0, 0, :label_len] = 0
        self.mask_full = self.mask_full.repeat(1,1,label_len+pred_len,1)
        self.k=indices.shape[-1]
        self.a = indices-1
        src = torch.zeros((B,label_len+pred_len), dtype=torch.bool,device=device)
        for i in range(label_len,label_len+pred_len):
            if i>label_len:
                a = indices[:,-self.k:]+1
                a[a>=label_len+pred_len]=label_len+pred_len-1
                indices=torch.cat((indices,a),dim=-1)
            for j in range(indices.shape[-1]):
                self.mask_full[:, 0, i]=self.mask_full[:, 0, i].scatter_(-1,indices[:,j].reshape(-1,1),src)
        self.mask_full = self.mask_full.to(device)

        self._mask = torch.ones((B,1,1,label_len+pred_len), dtype=torch.bool,device=device)
        self._mask[:, 0, 0, :label_len] = 0
        self._mask = self._mask.repeat(1,1,label_len+pred_len,1)
        self.label_len=label_len
        self.pred_len=pred_len

        self._mask = self._mask.to(device)

    @property
    def mask(self):
        return self._mask

    def forward(self):
        self.a = self.a+1
        self.a[self.a>=self.label_len+self.pred_len]=self.label_len+self.pred_len-1
        for j in range(self.k):
            print('a',self.a[:,j].shape)
            print('_mask',self._mask.shape)
            print('result',self._mask[:,:,self.a[:,j]].shape)
            self._mask[:,:,self.a[:,j]]=self.mask_full[:,:,self.a[:,j]]
        # print(self.a[0],self._mask[0,0])
        # print(self.a[0])
        return

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            # print((torch.triu(torch.ones(mask_shape), diagonal=1))[0])
            # print(self._mask[0])
            # print(self._mask.shape)
            # print(interd.shape)
            # self._mask = torch.logical_and(self._mask,interd)
            # print(self._mask[0])

            # scores.masked_fill_(attn_mask.mask, -np.inf)
            # self._mask = (1-torch.triu(torch.ones(mask_shape), diagonal=1)).bool().to(device)
            # print((torch.triu(torch.ones(mask_shape), diagonal=1))[0])
    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dytpe=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask