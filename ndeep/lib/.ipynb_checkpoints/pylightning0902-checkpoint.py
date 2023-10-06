
import os 
import torch
import torch.nn as nn 
import pandas as pd
import numpy as np
# import copy
import sys
sys.path.append('/home/petaon/python_packages/site-packages')
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
# import torch.utils.data as data
# from torch.utils.data import DataLoader, TensorDataset, random_split, SubsetRandomSampler
# from sklearn.model_selection import KFold
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# from sev_data_analysis import get_loader, evaluation_f, evaluation_mtl, lstm, mtl_lstm, loss_func, concordance_index
import pytorch_lightning as pl
from models import loss_func, concordance_index


import warnings
warnings.filterwarnings("ignore")

class LightninglstmClassifier(pl.LightningModule):

    def __init__(self, 
                input_size = 12, #n_feature, 
                layer_hidden_sizes = [10,20,15],
                num_layers = 3,
                bias = True,
                dropout = 0.0,
                bidirectional = False,
                batch_first = True,
                label_size = 1,
                E=1):
        super().__init__()
        self.E = E
        self.num_layers = num_layers
        self.rnn_models = nn.ModuleList([])
        if bidirectional:
            layer_input_sizes = [input_size] + [2 * chs for chs in layer_hidden_sizes]
        else:

            layer_input_sizes = [input_size] + layer_hidden_sizes
        for i in range(num_layers):
            self.rnn_models.append(nn.LSTM(input_size = layer_input_sizes[i],
                                     hidden_size = layer_hidden_sizes[i],
                                     num_layers = num_layers,
                                     bias = bias,
                                     dropout = dropout,
                                     bidirectional = bidirectional,
                                     batch_first = batch_first))
        self.label_size = label_size
        self.output_size = layer_input_sizes[-1]
        self.output_func = nn.Linear(self.output_size, self.label_size) 

    def forward(self, input_data):
        X = input_data['X'].float()
        M = input_data['M'].float()
        cur_M = input_data['cur_M'].float()
        _data = X
        for temp_rnn_model in self.rnn_models:
            _data, _ = temp_rnn_model(_data)
        outputs = _data
        all_output = outputs * M.unsqueeze(-1)
        n_batchsize, n_timestep, n_featdim = all_output.shape
        all_output = self.output_func(outputs.reshape(n_batchsize*n_timestep, n_featdim)).reshape(n_batchsize, n_timestep, self.label_size)
        cur_output = (all_output * cur_M.unsqueeze(-1)).sum(dim=1)
        return all_output, cur_output

    def surv_loss(self, pred, lifetime, event):
        return loss_func(pred, lifetime, event)

    def training_step(self, train_batch, batch_idx):
        # x, y = train_batch
        E = self.E
        _, yhat = self.forward(train_batch) #? self.model?
        loss = self.surv_loss(pred = yhat, 
                              lifetime=torch.tensor(np.sum(train_batch['T'].detach().cpu().numpy(), axis = 1)),
                               event=torch.tensor(train_batch['Y'][:, E-1]),
                               device=device)
        self.log('train_loss', loss)
        
        return {'loss': loss, 'pred': yhat, 'T': train_batch['T'], 'event': train_batch['Y'][:, E-1]}


    def validation_step(self, val_batch, batch_idx):
        E = self.E
        # x, y = val_batch
        _, yhat = self.forward(val_batch)
        loss = self.surv_loss(pred = yhat, 
                              lifetime=torch.tensor(np.sum(val_batch['T'].detach().cpu().numpy(), axis = 1)),
                               event=torch.tensor(val_batch['Y'][:, E-1]),
                               device=device)

        self.log('val_loss', loss) # 'val_acc',acc
        
    def test_step(self, test_batch, batch_idx):
        E = self.E
        _, yhat = self.forward(test_batch)
        acc = concordance_index(np.sum(test_batch['T'].detach().cpu().numpy(), axis = 1),  
                                np.exp(yhat), #? 
                                test_batch['Y'][:, E-1])
        self.log('c-index', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class Lightningmtl_lstm(pl.LightningModule):

    def __init__(self, 
                 in_features,
#                  batch_size,
#                  len_M = 192,
                 shared_layers = [3], #[, 3, 5],
                 num_tasks = 2, 
                 lstm_layers = [3, 3, 5],  
                 bias = True,
                 dropout = 0.0, 
                 batch_first = True,
                 bidirectional = False,
                 label_size = 1):
        super().__init__()
        self.sharedlayer = nn.Sequential(
            nn.Linear(in_features, shared_layers[0]* in_features),  
            nn.BatchNorm1d(shared_layers[0]* in_features),
            nn.ReLU(), 
            # nn.Dropout(p_dropout) 
        )

        if bidirectional:
            layer_input_sizes = [(shared_layers[-1]+1)*in_features] + [2 * chs for chs in lstm_layers]
        else:

            layer_input_sizes = [(shared_layers[-1]+1)*in_features] + lstm_layers #*(shared_layers[0]+1)*in_features

        self.rnn_models = nn.ModuleList([]) #temp

        for i in range(len(lstm_layers)):
            self.rnn_models.append(nn.LSTM(input_size = layer_input_sizes[i],
                                        hidden_size = lstm_layers[i],
                                        num_layers = len(lstm_layers),
                                        bias = bias,
                                        dropout = dropout,
                                        bidirectional = bidirectional,
                                        batch_first = batch_first))
        self.in_features = in_features
        self.shared_layers = shared_layers
        self.lstm_layers = lstm_layers
        self.label_size = label_size
        self.output_size = layer_input_sizes[-1]

        self.task = nn.ModuleList()
        for task in range(num_tasks):
            self.task.append(nn.Sequential(
                self.rnn_models,
                nn.Linear(self.output_size, self.label_size)))

        self.output_func = nn.Linear(self.output_size, self.label_size)
#         self.register_buffer('x_series', torch.zeros('batch_size',  #torch.empty
#                                len_M, #? how to soft code this dimension? (alter 192)
#                                (self.shared_layers[0]+1) * self.in_features))

    def forward(self, input_data): #
        x_wide = torch.stack([x[0] for x in input_data['X']])
        residual = x_wide.float() #0826 .to(device)
        shared = self.sharedlayer(x_wide.float()) #0826
        shared = torch.cat((shared, residual), dim=1) #concat
        time = np.sum(input_data['T'].detach().cpu().numpy(), axis = 1) # ? how about time = 1 -> 0 #0826
        X = []
        for i in range(len(shared)):
            if time[i]< 2:
                X.append(shared[i])
            else:
                X.append(shared[i].repeat(time[i].astype(int), 1))
        # Replicate x_series similar to DatasetReader
        x_series = torch.zeros(len(X),  #torch.empty
                               len(input_data['M'][0]), #? how to soft code this dimension? (alter 192)
                               (self.shared_layers[0]+1) * self.in_features)  #, device=device
#         x_series = self.x_series
        for i in range(len(X)):    
            x_series[i][:len(X[i]), :] = X[i] 
        X = x_series    #.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        M = input_data['M'].float()
        cur_M = input_data['cur_M'].float()
        output = []
        for task in self.task:
            _data = X   
            for temp_rnn_model in self.rnn_models: 
                _data, _ = temp_rnn_model(_data) #?
            outputs = _data
            all_output = outputs * M.unsqueeze(-1)
            n_batchsize, n_timestep, n_featdim = all_output.shape
            all_output = self.output_func(outputs.reshape(n_batchsize*n_timestep, n_featdim)).reshape(n_batchsize, n_timestep, self.label_size)
            cur_output = (all_output * cur_M.unsqueeze(-1)).sum(dim=1)
            output.append(cur_output) 
        return output 

    def surv_loss(self, pred, lifetime, event): #
        return loss_func(pred, lifetime, event, device=None) #device

    def training_step(self, train_batch, batch_idx): #, device
        task_nums = 2 #train_batch['Y'].shape[1] #[0]
        yhat = self.forward(train_batch) # , device
        train_loss = []
        for i in range(task_nums):
            loss = self.surv_loss(pred = yhat[i], 
                                lifetime=torch.tensor(np.sum(train_batch['T'].detach().cpu().numpy(), axis = 1)), #to device?
                                event=torch.tensor(train_batch['Y'][:, i]),
                                ) #device=device
            train_loss.append(loss)
        train_loss =  sum(train_loss)/len(train_loss)
        self.log('train_loss', train_loss) #list value cannot be logged
        
        return {'loss': train_loss, 'pred': yhat, 'T': train_batch['T'], 'event': train_batch['Y'][:, i]}


    def validation_step(self, val_batch, batch_idx): #, device
        task_nums = 2 #val_batch['Y'].shape[1] #[0]
        yhat = self.forward(val_batch) #, device
        val_loss = []
        for i in range(task_nums):
            loss = self.surv_loss(pred = yhat[i], 
                                lifetime=torch.tensor(np.sum(val_batch['T'].detach().cpu().numpy(), axis = 1)), #to device?
                                event=torch.tensor(val_batch['Y'][:, i]),
                                ) #device=device
            val_loss.append(loss)
        val_loss =  sum(val_loss)/len(val_loss)
        self.log('val_loss', val_loss)
        
    def test_step(self, test_batch, batch_idx): #, device
        task_nums = 2 #test_batch['Y'].shape[1] #[0]
        yhat = self.forward(test_batch) #, device
        accs = {}
        for i in range(task_nums):
            acc = concordance_index(np.sum(test_batch['T'].detach().cpu().numpy(), axis = 1),  
                                    np.exp(yhat[i]), #? 
                                    test_batch['Y'][:, i])
            accs['event_' + str(i+1)] = acc
            # # self.log(globals()['c-index_' + str(i+1)], acc) #accs
        self.log('c-index', accs) # accs Problem 0828, why acc <0.01

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def pl_lstm(train, train_f, test, valid, input_size, n_epochs=1):
    model = LightninglstmClassifier(input_size = input_size, #n_feature, 
                    layer_hidden_sizes = [3,3,5])
    trainer = pl.Trainer(max_epochs=n_epochs, gpus='1') #
    trainer.fit(model, train, valid)
    test_result = trainer.test(model, dataloaders=test, verbose=False)
    train_result = trainer.test(model, dataloaders=train_f, verbose=False)
    return train_result, test_result

def pl_mtl_lstm(train, train_f, test, valid, input_size, n_epochs=1):
    model = Lightningmtl_lstm(in_features=input_size)
    trainer = pl.Trainer(max_epochs=n_epochs)  #, profiler="simple" gpus='1'
    trainer.fit(model, train, valid) #?device
    train_result = trainer.test(model, dataloaders=train_f, verbose=False)
    test_result = trainer.test(model, dataloaders=test, verbose=False)
    return train_result, test_result