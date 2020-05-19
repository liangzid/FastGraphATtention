import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import time
# self-define
# import lsh.lsh_mips as lsh

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

    def __init__(self, in_features, out_features, dropout, alpha, concat=True,seed=3933,cuda=True,bucket=4):
        super(LSHAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.seed=seed
        self.cuda=cuda
        self.bucket=bucket
        
        # some infomation for LSH config
        self.nHashTable=1
        self.nbuckets=4

        self.kW = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.kW.data, gain=1.414) # one special init method,by  Bengio.
        
        self.vW = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.vW.data, gain=1.414) # one special init method,by  Bengio.
        
        # self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # t1=time.time()
        kh = torch.mm(input, self.kW) # mm -> matrix multiplication 
        N,hiddenSize = kh.shape
        
        # t2=time.time()
        vh =torch.mm(input,self.vW)
        # t3=time.time()
        # h.repeat(x,y,z) means expand x,y,z 倍 in each dimension respectivly.
        # shape of a_input: [N,N,2*8]
        
        # t4=time.time()         
        nHashTable=1
        bucketSize=self.bucket
        if self.cuda:                                                  
            rotations_rands=torch.randn(nHashTable,hiddenSize,bucketSize//2).cuda() #[nhashtable,hiddenSize,bucketSize//2]
        else:
            rotations_rands=torch.randn(nHashTable,hiddenSize,bucketSize//2)
        # t5=time.time()
        rota_vectors=torch.matmul(1.0*kh,rotations_rands)                #[nhashtable,N,bucketsize//2]
        # t6=time.time()
        rota_vectors=torch.cat([rota_vectors,-rota_vectors],-1)           #[nhashtable,N,bucketsize]
        buckets=torch.argmax(rota_vectors,dim=2)                       #[nhashtable,N]
        # 此时返回的是索引，这个索引可以理解成是bucket，代表了不同节点的bucket
        # print(buckets)
        # t7=time.time()
        #-------------------------------need to be optimalize-------------------------
        values,indexes=buckets.sort(dim =1)                              #[nhashtable]
        # 这里的value就是上面的bucket，这里的indexes就是排序之后的value在之前位置的index
        #-----------------------------------------------------------------------------
        # t8=time.time()
        # to x class
        values=values.tolist()[0]
        indexes=indexes.tolist()[0]
        # print(indexes)
        # t9=time.time()
        tempList=[]
        indexList=[]

        # print(values)

        for i in range(bucketSize):
            if i in values:
                tempList.append(values.index(i)) # 这个循环的意思是：对于每个bucket（也就是i），都需要找到相对应的开始index。
        # t11=time.time()
        for i in range(len(tempList)):
            if i != len(tempList)-1:
                # print(i)
                indexList.append(indexes[tempList[i]:tempList[i+1]])
            else:
                indexList.append(indexes[tempList[i]:])
        # print(indexList)
        # t12=time.time()
        del tempList        
        # t13=time.time()
        if self.cuda:
            K=-9e15*torch.ones((N,N)).cuda()
        else:
            K=-9e15*torch.ones((N,N))
        #print(indexList)
        # print('=='*10+'\n'+str(N)+'\n\n')
        for x in indexList:
            # print(len(x))
            
            K[torch.tensor(x).repeat(len(x)),torch.tensor(x).repeat(len(x),1).T.reshape(1,-1).view(-1)]=torch.mm(kh[x,:],kh[x,:].T).view(-1)/math.sqrt(hiddenSize)
        # t14=time.time()
        # print(values,indexes)
        # rotations_rand=np.random.randn(hiddenSize,bucketSize//2,seed=self.seed)
        # rota_vectors=np.dot(h,rotations_rand)
        # rota_vectors=np.hstack([rota_vectors,-rota_vectors])
        # buckets=np.argmax(rota_vectors,axis=-1)
        
        # K=torch.mm(kh,kh.T)
        # K=K/(math.sqrt(hiddenSize))
        # t15=time.time()
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

        if self.cuda:

            zero_vec = -9e15*torch.ones_like(K).cuda()
        else:
            zero_vec = -9e15*torch.ones_like(K)
        # print(adj.shape)
        # print(K.shape)
        attention = torch.where(adj > 0, K, zero_vec) # which means if here exists a edge,use e' element,else: use zero_vec' element(not zero beacuse softmax next).
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention,vh)

        # print(t2-t1)
        # print(t3-t2)
        # print(t4-t3)
        # print(t5-t4)
        # print(t7-t6)
        # print(t8-t7)
        # print(t9-t8)
        # print(t11-t9)
        # print(t12-t11)
        # print(t13-t12)
        # print(t14-t13)
        # print(t15-t14)
        # # print(t16-t15)
        # # print(t17-t16)

        if self.concat:
            return F.elu(h_prime) # avtivate or not.
        else:
            return h_prime

    def __repr__(self): # representatioin
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
