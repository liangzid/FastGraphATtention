from dgl.data import MiniGCDataset as md
import scipy as sci
import torch
import numpy as np

# # generate GCdata.
# N=10
# nTrain=3
# nValidate=10
# nTest=20

def generate(num=50,Nmin=10,Nmax=100):
    dataset=md(num,Nmin,Nmax)

    # permutation
    permu=np.random.permutation(np.arange(len(dataset)))
    graphs=[]
    labels=[]
    adj = []
    for i in range(len(dataset)): 
        g,l=dataset[permu[i]]
        graphs.append(g)
        labels.append(l)
        adj.append(g.adjacency_matrix().to_dense().numpy())

    path='../data/sdata/'

    if num==1:
        i=0
        print(graphs[0])
        # features=graphs[i].in_degrees().float().view(-1,1)
        features=np.random.randint(0,10,(10000,2000)).astype(float)
        # print(features.shape)
        np.save(path+'sdata_feature_{}.npy'.format(i),features)
        np.save(path+'sdata_adj_{}.npy'.format(i),adj[i])

    else:
        # hidden layer feature.
        features = [graphs[i].in_degrees().float().tolist() for i in range(len(graphs))]
        # print(type(features))
        
 
        for i in range(len(dataset)):
            np.save(path+'sdata_feature_{}.npy'.format(i),features[i])
            np.save(path+'sdata_adj_{}.npy'.format(i),adj[i])
    np.save(path+'sdata_label.npy',np.array(labels))

    # with open('sdata_features.txt', 'w') as file:  
    #     for f in features:
    #         file.write(str(f))
    #         file.write('\n')
    # file.close()
    # with open('sdata_adj.txt', 'w') as file:  
    #     for a in adj:
    #         file.write(str(a))
    #         file.write('\n')
    # file.close()

    # with open('sdata_label.txt', 'w') as file: 
    #     for l in labels:
    #         file.write(str(l))
    #         file.write('\n')
    # file.close()

    # features=np.array(features).view(len(graphs),-1)
    # np.save('sdata_features.npy',features)
    # np.save('sdata_adj.npy',adj)
    # np.save('sdata_label.npy',np.array(labels))
    print('done.')
    print('generate {0} graph,to{1}'.format(num,path))


# read data.
def read():
    path='../data/sdata/'
    # ndataset=30
    features,adjs=[],[]
    labels=np.load(path+'sdata_label.npy')
    ndataset=len(labels)
    for i in range(ndataset):
        features.append(np.load(path+'sdata_feature_{}.npy'.format(i)))
        adjs.append(np.load(path+'sdata_adj_{}.npy'.format(i)))
    # print(labels)

    # print(features)
    # print(adjs)
    # print(labels)
    return features,adjs,labels


    # with open('sdata_features.txt', 'r') as file:

    #     features=[[float(element.strip()) for element in f_list] for f_list in file]
    #     print(features)
    # file.close()
    # with open('sdata_adj.txt', 'r') as file:  
    #     for a in adj:
    #         file.write(str(a))
    #         file.write('\n')
    # file.close()

    # with open('sdata_label.txt', 'r') as file: 
    #     for l in labels:
    #         file.write(str(l))
    #         file.write('\n')
    # file.close()

if __name__=="__main__":
    generate(num=1,Nmin=10000,Nmax=10001)
    # generate()
    

