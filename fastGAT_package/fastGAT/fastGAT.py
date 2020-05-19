import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from fastGAT.utils import load_data, accuracy
from fastGAT.model import GAT,FastGAT


class exe(): 
    def __init__(self,save_model=True,datasetpath='fastGAT/data/cora/cora',no_cuda=False,bucket=4,fastmode=False,seed=3933,epochs=200,lr=0.005,weight_delay=5e-4,hidden=8,nb_heads=8,dropout=0.6,alpha=0.2,patience=100,fastGAT=1):


        self.fastmode=fastmode
        self.seed=seed
        self.epochs=epochs
        # self.lr=lr
        # self.weight_delay=weight_delay
        self.patience=patience
        self.fastGAT=fastGAT
        self.save_model=save_model
        self.datasetpath=datasetpath
        self.bucket=bucket

        self.cuda=not no_cuda and torch.cuda.is_available()

        self.best_epoch_during_training=-1

        self.save_name='.'+str(self.fastGAT)+'.pkl'

        # init seed.
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.cuda:
            torch.cuda.manual_seed(self.seed)

        # load data
        adj, features, labels, self.idx_train, self.idx_val, self.idx_test = load_data(datasetpath)        

        # load model
        if fastGAT==0:
            # Model and optimizer
            self.model = GAT(nfeat=features.shape[1], 
                        nhid=hidden, 
                        nclass=int(labels.max()) + 1, 
                        dropout=dropout, 
                        nheads=nb_heads, 
                        alpha=alpha)
            self.optimizer = optim.Adam(model.parameters(), 
                                lr=lr, 
                                weight_decay=weight_decay)
        else:
            # Model and optimizer
            self.model = FastGAT(nfeat=features.shape[1], 
                        nhid=hidden, 
                        nclass=int(labels.max()) + 1, 
                        dropout=dropout, 
                        nheads=nb_heads,
                        bucket=self.bucket, 
                        alpha=alpha)
            self.optimizer = optim.Adam(self.model.parameters(), 
                                lr=lr, 
                                weight_decay=weight_delay)


        if self.cuda:
            self.model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()
            self.idx_test = self.idx_test.cuda()

        self.features, self.adj, self.labels = Variable(features), Variable(adj), Variable(labels)    


    def tra_val(self):
        model=self.model
        optimizer=self.optimizer
        # Train model
        t_total = time.time()
        loss_values = []
        bad_counter = 0
        best = self.epochs + 1
        best_epoch = 0
        save_name=self.save_name

        #=====================================per epoch===========================================
        for epoch in range(self.epochs):
            t = time.time()
            # model.train()
            optimizer.zero_grad()
            output = model(self.features, self.adj)
            loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
            acc_train = accuracy(output[self.idx_train], self.labels[self.idx_train])
            loss_train.backward()
            optimizer.step()

            if not self.fastmode:
                # Evaluate validation set performance separately,
                # deactivates dropout during validation run.
                model.eval()
                output = model(self.features, self.adj)

            loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
            acc_val = accuracy(output[self.idx_val], self.labels[self.idx_val])
            print('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(loss_train.data.item()),
                'acc_train: {:.4f}'.format(acc_train.data.item()),
                'loss_val: {:.4f}'.format(loss_val.data.item()),
                'acc_val: {:.4f}'.format(acc_val.data.item()),
                'time: {:.4f}s'.format(time.time() - t))

            loss_values.append(loss_val.data.item())

            torch.save(model.state_dict(), ('{}'+save_name).format(epoch))
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == self.patience:
                break

            files = glob.glob('*'+save_name)
            for file in files:
                epoch_nb = int(file.split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(file)

        files = glob.glob('*'+save_name)
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(file)

        print("Training Finished.")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print('Loading {}th epoch'.format(best_epoch))

        self.best_epoch_during_training=best_epoch
        
        print('OK.')



    def test(self):


        print("make sure you have train the model. or test will be failed.")

        # Reload best model
        model=self.model
        model.load_state_dict(torch.load(('{}'+self.save_name).format(self.best_epoch_during_training)))

        # Testing
        model.eval()
        output = model(self.features, self.adj)
        loss_test = F.nll_loss(output[self.idx_test], self.labels[self.idx_test])
        acc_test = accuracy(output[self.idx_test], self.labels[self.idx_test])
        print("Test set results:",
            "loss= {}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()))
        print("done.")
