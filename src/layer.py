import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
# self-define
import lsh.lsh_mips as lsh

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414) # one special init method,by  Bengio.
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W) # mm -> matrix multiplication 
        N = h.size()[0]

        # h.repeat(x,y,z) means expand x,y,z 倍 in each dimension respectivly.
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # print("a_input shape:{}".format(a_input.shape))
        # [N*N,2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # [N,N]
        # torch.matmul -> multiplication in tensors. (4,5,6)*(2,5) -> (4,2,6)
        # torch.squeeze() needs take the size 1 in tensor dimension out. : if the 2th dimension is size 1,then take it out.
        
        # e.shape ->N,N

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec) # which means if here exists a edge,use e' element,else: use zero_vec' element(not zero beacuse softmax next).
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime) # avtivate or not.
        else:
            return h_prime

    def __repr__(self): # representatioin
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class LSHAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(LSHAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        # self.seed=seed

        self.kW = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.kW.data, gain=1.414) # one special init method,by  Bengio.
        
        self.vW = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.vW.data, gain=1.414) # one special init method,by  Bengio.
        
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        kh = torch.mm(input, self.kW) # mm -> matrix multiplication 
        N,hiddenSize = kh.shape

        vh =torch.mm(input,self.vW)
        # h.repeat(x,y,z) means expand x,y,z 倍 in each dimension respectivly.
        # shape of a_input: [N,N,2*8]
        '''
        bucketSize=N//2
        rotations_rand=np.random.randn(hiddenSize,bucketSize//2)
        rota_vectors=np.dot(h,rotations_rand)
        rota_vectors=np.hstack([rota_vectors,-rota_vectors])
        buckets=np.argmax(rota_vectors,axis=-1)
        '''
        K=torch.mm(kh,kh.T)
        K=K/(math.sqrt(hiddenSize))
        # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # print(type(a_input))
        # a_data=a_input.view(-1,2*self.out_features)
        # candidate,index=lsh.lsh_K_MIPS(K=10,L=2,datas=a_data.cpu().detach().numpy().tolist(),
        # querys=self.a.cpu().detach().numpy().tolist(),
        # num_neighbours=-1,rand_range=3,needInformation=False)
        # x,_=a_input.shape
        # result=torch.ones(a_input.shape)*(-9e15)
        # result[index]=candidate.reshape(N,N,len(index))
        # e = self.leakyrelu(torch.matmul(result, self.a).squeeze(2))
        # torch.matmul -> multiplication in tensors. (4,5,6)*(2,5) -> (4,2,6)
        # torch.squeeze() needs take the size 1 in tensor dimension out. : if the 2th dimension is size 1,then take it out.
        
        # e.shape -> N,N/2

        zero_vec = -9e15*torch.ones_like(K)
        attention = torch.where(adj > 0, K, zero_vec) # which means if here exists a edge,use e' element,else: use zero_vec' element(not zero beacuse softmax next).
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention,vh)

        if self.concat:
            return F.elu(h_prime) # avtivate or not.
        else:
            return h_prime

    def __repr__(self): # representatioin
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
