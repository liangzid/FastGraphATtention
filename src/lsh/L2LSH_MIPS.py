# -*- coding: utf-8 -*-
'''
-----------------------------------------------------------------------------------------------
This interface is based on [LSH_MIPS](https://github.com/jfpu/lsh_mips)

the changes is:

1) suitable for python 3.x
2) more useful and easier to use, especially for K-MIPS
3) sovle the problem about the test accuracy in lsh_tester.py

                        liangzid,3,17,2020.
-----------------------------------------------------------------------------------------------
'''


"""
# Locality Sensitive Hashing for MIPS

## Reference:

- LSH:
[Locality-Sensitive Hashing Scheme Based on p-Stable Distributions](http://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/p253-datar.pdf)

- L2-ALSH:
[Asymmetric LSH (ALSH) for Sublinear Time Maximum Inner Product Search (MIPS)](https://papers.nips.cc/paper/5329-asymmetric-lsh-alsh-for-sublinear-time-maximum-inner-product-search-mips.pdf)

- Sign-ALSH(Cosine):
[Improved Asymmetric Locality Sensitive Hashing (ALSH) for Maximum Inner Product Search (MIPS)](http://auai.org/uai2015/proceedings/papers/96.pdf)

- Simple-ALSH(Cosine):
[On Symmetric and Asymmetric LSHs for Inner Product Search](https://arxiv.org/abs/1410.5518)

"""

import random
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from operator import itemgetter

from lsh import *
from lsh_wrapper import *
from lsh_tester import *
from global_mgr import gol

def lsh_K_MIPS(K,L,datas,querys,num_neighbours=-1,rand_range=1,needInformation=False):
    '''
    ----------------------------------------------------------------------
         Finding the K Maximumal Inner Product Vectors of query.
         Return is a list-list which for each query have a list
         to store the vectors candidate.
    ----------------------------------------------------------------------
    ======================================================================
    K: num of hashFunction of each hash table;
    L: num of hash table;
    datas: vectors of candidate. --size--:numofData*vectorDimension
    querys: query vector. --size: vectorDimension
    num_neighbours: num of vectors to be found 
    rand_range: the norm
    needInformation: DEBUG(True) of Release(False)
    ======================================================================
    '''

    kdata = len(datas[0])
    qdata = len(querys[0])

    num_data=len(datas)
    d=len(datas[0])
    if num_neighbours==-1: # auto generate the nums of neighbours
        num_neighbours=int(0.5*num_data)
    
    if needInformation:
        tester=LshTesterFactory.createTester(qtype='simple',mips=True,datas=datas,queries=querys,rand_num=1,
        num_neighbours=num_neighbours)
        tester.run(k_vec=[2,4,8,16,32],l_vec=[2,4,8,16,32])
        return "--------------------- test END ---------------------------"
    
    # m: ALSH extend metrix length. default 1
    m = 1

    # datas & queries transformation
    dratio, dmax_norm, norm_datas = g_transformation(datas)
    norm_queries = g_normalization(querys)

    assert (kdata == len(norm_datas[0]))
    assert (qdata == len(norm_queries[0]))
    assert (len(datas) == len(norm_datas))
    assert (len(querys) == len(norm_queries))

    # expand k dimension into k+2m dimension
    ext_datas = g_index_simple_extend(norm_datas, m)
    ext_queries = g_query_simple_extend(norm_queries, m)
    new_len = d + 2 * m


    validate_metric = np.dot
    compute_metric = L2Lsh.distance
    # exact_hits = [[ix for ix, dist in self.linear(q, validate_metric, num_neighbours)] for q in querys]
    # print('==============================')
    # print('SimpleAlshTester ' + ' TEST:')
    # print('L\tk\tacc\ttouch')

    # concatenating more hash functions increases selectivity
    lsh = LshWrapper(lsh_type='cosine', d=len(ext_datas[0]), r=rand_range, k=K, L=L)
    lsh.index(ext_datas)
    all_vectors=[]
    for q in ext_queries:    
        lsh_hits = [ix for ix, dist in lsh.query(q, compute_metric, num_neighbours)]
        #--------------------------------------------------------------------------
        vector=np.array(datas)[lsh_hits]
        # print(len(vector))
        # print(vector)
        # -------------------------------------------------------------------------
        all_vectors.append(vector)
        # print(vector.shape)
    
    return all_vectors
    
    


if __name__ == "__main__":

    # parameters initializing ......
    radius = 0.3
    r_range = 10 * radius
    d = 100
    xmin = 0
    xmax = 10
    num_datas = 1000
    num_queries = 10
    datas = [[random.randint(xmin, xmax) for i in range(d)] for j in range(num_datas)]

    queries = []
    for point in datas[:num_queries]:
        queries.append([x + random.uniform(-radius, radius) for x in point])

    gol._init()
    #gol.set_value('DEBUG', True)

    # function calling ......
    result=lsh_K_MIPS(K=10,L=10,datas=datas,querys=queries,num_neighbours=-1,rand_range=3,needInformation=False)
    # result=lsh_K_MIPS(K=10,L=10,datas=datas,querys=queries,num_neighbours=-1,rand_range=3,needInformation=True)    
    # print(len(result))
    # print(len(result[0]))
    # print(len(result[0][0]))