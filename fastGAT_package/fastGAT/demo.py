from fastGAT.fastGAT import exe

procedure=exe(save_model=True,datasetpath='fastGAT/data/cora/cora',no_cuda=False,
            fastmode=False,seed=3933,epochs=200,lr=0.005,weight_delay=5e-4,hidden=8,nb_heads=4,
            bucket=2,dropout=0.6,alpha=0.2,patience=100,fastGAT=1)
    
procedure.tra_val()

procedure.test()