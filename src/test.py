import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np


# test LSH
kh= torch.tensor([[1,2,3,4,5,6,7,8],[3,3,5,6,3,3,1,2],[1,2,3,4,5,6,7,7],[1,1,2,3,4,5,6,7],
[1,1,2,4,5,6,7,8],[3,3,5,6,3,3,2,2]   
])     # 6*8 [N,hiddenSize]

N,_=kh.shape
hiddenSize=8
nHashTable=1

bucketSize=4                                                  
rotations_rands=torch.randn(nHashTable,hiddenSize,bucketSize//2) #[nhashtable,hiddenSize,bucketSize//2]
rota_vectors=torch.matmul(1.0*kh,rotations_rands)                #[nhashtable,N,bucketsize//2]
# print(rota_vectors.shape) #[4,6,2]
rota_vectors=torch.cat([rota_vectors,-rota_vectors],-1)           #[nhashtable,N,bucketsize]
# print(rota_vectors.shape) #[4,12,2]
buckets=torch.argmax(rota_vectors,dim=2).squeeze()                       #[nhashtable,N]
# print(buckets.shape)
# print(buckets)
values,indexes=buckets.sort()                              #[nhashtable]
indexList=[]
# print(indexes)

# values=torch.tensor([1,1,1,2,2])
# indexes=torch.tensor([0,1,3,2,4])

values=values.tolist()
indexes=indexes.tolist()
print(indexes)
tempList=[]
for i in range(bucketSize):
    if i in values:
        tempList.append(values.index(i))

for i in range(len(tempList)):
    if i != len(tempList)-1:
        indexList.append(indexes[tempList[i]:tempList[i+1]])
    else:
        indexList.append(indexes[tempList[i]:])

print(indexList)
del tempList