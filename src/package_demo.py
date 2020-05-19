import fastGAT as ft
==================================================================================================
                                        “训练过程”示例
==================================================================================================

# 定义数据集路径
datasetPath='../data/cora/cora'

# 设置模型的执行参数，从而对训练和测试流程进行定义
procedure=exe(save_model=True,datasetpath=datasetPath,no_cuda=False,
            fastmode=False,seed=3933,epochs=200,lr=0.005,weight_delay=5e-4,hidden=8,nb_heads=4,
            bucket=2,dropout=0.6,alpha=0.2,patience=100,fastGAT=1)
    
# 执行训练过程与验证过程
procedure.tra_val()

# 执行测试过程
procedure.test()

===================================================================================================
                                        “模型调用”示例
===================================================================================================

ourmethod=FastGAT(nfeat=features.shape[1], # 设置特征向量维度 
                        nhid=hidden,       # 设置隐藏层维度
                        nclass=int(labels.max()) + 1, # 设置分类个数
                        dropout=dropout,              # 设置dropout率
                        nheads=nb_heads,              # 设置注意力头的个数
                        bucket=self.bucket,           # 设置降类哈希桶的个数
                        alpha=alpha)                  # 设置leaky_relu参量

