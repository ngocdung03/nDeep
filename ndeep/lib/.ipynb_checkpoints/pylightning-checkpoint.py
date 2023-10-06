import lifelines
import torch
from torch import nn
import numpy as np
import time

import pytorch_lightning as pl
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

    def surv_loss(self, pred, lifetime, event, device):
        return loss_func(pred, lifetime, event, device)

    def training_step(self, train_batch, batch_idx):
        # x, y = train_batch
        E = self.E
        _, yhat = self.forward(train_batch) #? self.model?
        loss = self.surv_loss(pred = yhat, 
                              lifetime=torch.tensor(np.sum(train_batch['T'].detach().numpy(), axis = 1)),
                               event=torch.tensor(train_batch['Y'][:, E-1]),
                               device=device)
        self.log('train_loss', loss)
        
        return {'loss': loss, 'pred': yhat, 'T': train_batch['T'], 'event': train_batch['Y'][:, E-1]}


    def validation_step(self, val_batch, batch_idx):
        E = self.E
        # x, y = val_batch
        _, yhat = self.forward(val_batch)
        loss = self.surv_loss(pred = yhat, 
                              lifetime=torch.tensor(np.sum(val_batch['T'].detach().numpy(), axis = 1)),
                               event=torch.tensor(val_batch['Y'][:, E-1]),
                               device = device)

        self.log('val_loss', loss) # 'val_acc',acc
        
    def test_step(self, test_batch, batch_idx):
        E = self.E
        _, yhat = self.forward(test_batch)
        acc = concordance_index(np.sum(test_batch['T'].detach().numpy(), axis = 1),  
                                np.exp(yhat), #? 
                                test_batch['Y'][:, E-1])
        self.log('c-index', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# data
train_dataloader = rtrain_reader
val_loader = rvalid_reader
test_loader = rtest_reader

# train
model = LightninglstmClassifier(input_size = 29, #n_feature, 
                layer_hidden_sizes = [3,3,5])
trainer = pl.Trainer(max_epochs=1) #

trainer.fit(model, train_dataloader, val_loader)
test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
test_result