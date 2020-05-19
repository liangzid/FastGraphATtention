import torch
import torch.nn as nn
import torch.nn.functional as F
from  fastGAT.layer import GraphAttentionLayer,LSHAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

class FastGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,bucket=4,cuda=True):
        super(FastGAT, self).__init__()
        self.dropout = dropout
        self.layernorm=nn.LayerNorm(nhid*nheads)

        self.attention1 = [LSHAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha,bucket=bucket,cuda=cuda, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attention1):
            self.add_module('attention1_{}'.format(i), attention)
        self.inter_att=LSHAttentionLayer(nhid*nheads,nfeat,dropout=dropout,alpha=alpha,concat=False,bucket=bucket,cuda=cuda)
        #'''
        self.attention2 = [LSHAttentionLayer(nhid*nheads, nhid, dropout=dropout, alpha=alpha,bucket=bucket,cuda=cuda, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attention2):
            self.add_module('attention2_{}'.format(i), attention)
        #'''
        self.out_att = LSHAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False,bucket=bucket,cuda=cuda)

    def forward(self,x,adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = torch.cat([att(x, adj) for att in self.attention1], dim=1)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        #print("x:{0},\n,x1:{1}\n".format(x.shape,x1.shape))
        
        #lone=self.layernorm(x1+x)
        #x2 = F.elu(self.inter_att(lone,adj))
        #print(x2.shape)
        #print(x.shape)
        #x2 = F.dropout(x2, self.dropout, training=self.training)
        #x2 = torch.cat([att(x2,adj) for att in self.attention2],dim=1)
        #x2 = F.dropout(x2, self.dropout, training=self.training)
        x = F.elu(self.out_att(x1, adj))
        return F.log_softmax(x, dim=1)


