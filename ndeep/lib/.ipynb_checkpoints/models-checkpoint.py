import lifelines
import torch
from torch import nn
import numpy as np
import time
import pycox
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
import torchtuples as tt
import pytorch_lightning as pl

device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def onePair(x0, x1):
    c = np.log(2.)
    m = nn.LogSigmoid() 
    return 1 + m(x1-x0) / c

def rank_loss(pred, obs, delta, epsilon, device=None):
    N = pred.size(0)
    allPairs = onePair(pred.view(N,1), pred.view(1,N))

    temp0 = obs.view(1, N) - obs.view(N, 1)
    # indices based on obs time
    temp1 = temp0>0
    # indices of event-event or event-censor pair
    temp2 = delta.view(1, N) + delta.view(N, 1)
    temp3 = temp2>0
    # indices of events
    temp4 = delta.view(N, 1) * torch.ones(1, N, device=device) if device else delta.view(N, 1) * torch.ones(1, N)
    # selected indices
    final_ind = temp1 * temp3 * temp4
    out = allPairs * final_ind
    return out.sum() / (final_ind.sum() + epsilon)

def mse_loss(pred,  obs, delta):
    mse = delta*((pred - obs) ** 2)

    ind = pred < obs
    delta0 = 1 - delta
    p = ind * delta0 * (obs - pred)**2 
    return mse.mean(), p.mean()

def loss_func(pred, lifetime, event, device, lambda1 = 1, lambda2 = 0.2, epsilon = 1e-3): #, device=device
    mseloss, penaltyloss = mse_loss(pred, lifetime.unsqueeze(1), event.unsqueeze(1))
    rankloss = rank_loss(pred, lifetime.unsqueeze(1), event.unsqueeze(1), epsilon, device) #, device
    loss = mseloss + lambda1*penaltyloss - lambda2*rankloss
    return loss
    
def concordance_index(event_times, predicted_scores, event_observed=None) -> float:
    event_times, predicted_scores, event_observed = lifelines.utils.concordance._preprocess_scoring_data(event_times, predicted_scores, event_observed)
    num_correct, num_tied, num_pairs = lifelines.utils.concordance._concordance_summary_statistics(event_times, predicted_scores, event_observed)
    num_pairs += 1e-50 
    return lifelines.utils.concordance._concordance_ratio(num_correct, num_tied, num_pairs)

class lstm(nn.Module):
    def __init__(self,
                 input_size,
                 layer_hidden_sizes = [3,3,5], #10,20,15
                 num_layers = 3,
                 bias = True,
                 dropout = 0.0,
                 bidirectional = False,
                 batch_first = True,
                 label_size = 1):
        super(lstm, self).__init__()

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
        self.output_func = nn.Linear(self.output_size, self.label_size) # what happen when label_size > 1 ?

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
        return cur_output
    
class mtl_deephit(nn.Module):
    def __init__(self, in_features, task_nums, hidden_layers = [3, 3, 5], out_features=[1, 1], p_dropout=0.6):
        super().__init__()
        self.sharedlayer = nn.Sequential(
            nn.Linear(in_features, hidden_layers[0]* in_features),  
            nn.BatchNorm1d(hidden_layers[0]* in_features),
            nn.ReLU(), 
            # nn.Dropout(p_dropout) 
        ) 
        self.task = nn.ModuleList()
        for task in range(task_nums):
            self.task.append(nn.Sequential(
                    nn.Linear(hidden_layers[0]* in_features + in_features, hidden_layers[1]*in_features),
                    nn.BatchNorm1d(hidden_layers[1]*in_features),
                    nn.ReLU(),
                    # nn.Dropout(p_dropout),
                    nn.Linear(hidden_layers[1]*in_features, hidden_layers[2]*in_features),
                    nn.BatchNorm1d(hidden_layers[2]*in_features),
                    nn.ReLU(),
                    nn.Dropout(p_dropout),
                    nn.ReLU(),  #
                    nn.Linear(hidden_layers[2]*in_features, out_features[0]),
        )
        )
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain = nn.init.calculate_gain('relu'))

    def forward(self, x):
        residual = x
        shared = self.sharedlayer(x)
        # Residual concatenating
        shared = torch.cat((shared, residual), dim=1) 
        output = []
        for task in self.task:  #self.task[-1]
        	output.append(task(shared))
        return output
    
class mtl_lstm(nn.Module):
    def __init__(self, 
                 in_features, 
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

    def forward(self, input_data, device): #0826
        x_wide = torch.stack([x[0] for x in input_data['X']])
        residual = x_wide.float().to(device) #0826
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
                               (self.shared_layers[0]+1) * self.in_features)
        for i in range(len(X)):    
            x_series[i][:len(X[i]), :] = X[i] 
        X = x_series.to(device) #0826
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

def evaluate_lstm(model, train_loader, train_loader_f, test_loader, learning_rate=1e-3, n_epochs=3, loss_func=loss_func, E=1, device=device2):
    torch.manual_seed(1)
#    model.to('cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
    size = len(train_loader.dataset)
    model.to(device)
    # model.train() ##
    # training model
    for i in range(1, n_epochs+1):
        st = time.time()
        model.train() ##
        losses = []
        p1 = time.time()
        for n, batch in enumerate(train_loader):
            optimizer.zero_grad()
            y_hat = model(batch)
            loss = loss_func(pred=y_hat, 
                             lifetime=torch.tensor(np.sum(batch['T'].numpy(), axis=1)), 
                             event=torch.tensor(batch['Y'][:, E-1]),device = device)
            losses.append(loss)
            loss.backward()
            optimizer.step()
            if n > 0 and n % 20 == 0:
                print('[%d/%d] -'%((n+1)*len(batch['X']), size), 'loss:', loss.item(), round(time.time()-p1, 4), 's')
                p1 = time.time()
        print('%dth epoch: loss(%f) - %f s' % (i, sum(losses)/len(losses), round(time.time()-st, 4)))
    
    # evaluate with trainset
    y_pred = []
    with torch.no_grad():
        model.eval()
        [y_pred.append(model(t)) for t in train_loader_f]
        t = [t['T'].numpy() for t in train_loader_f]
        tt = [t['Y'][:, E-1] for t in train_loader_f]
    y_pred = sum([x.squeeze().tolist() for x in y_pred], [])
    train_c = concordance_index(np.sum(np.concatenate(t), axis=1), np.exp(y_pred), np.concatenate(tt))

    # evaluate with testset
    y_pred = []
    with torch.no_grad():
        model.eval()
        [y_pred.append(model(t)) for t in test_loader]
        t = [t['T'].numpy() for t in test_loader]
        tt = [t['Y'][:, E-1] for t in test_loader]
    y_pred = sum([x.squeeze().tolist() for x in y_pred], [])
    test_c = concordance_index(np.sum(np.concatenate(t), axis=1), np.exp(y_pred), np.concatenate(tt))
    return train_c, test_c

def evaluation_f(model_instance, train_loader, rtrain_pred, test_loader, valid_loader = None,
               learning_rate= 1e-3, n_epochs=2, loss_func=loss_func, 
               E = 1, c_index=concordance_index, device=device2):

    torch.manual_seed(1)
    model = model_instance #lstm(input_size=29, layer_hidden_sizes = [3, 3, 5])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-8)

    # Training
    for i in range(1, n_epochs+1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            cur_out = model(batch)  
            loss1 = loss_func(pred = cur_out, 
               lifetime=torch.tensor(np.sum(batch['T'].detach().cpu().numpy(), axis = 1)).to(device),  ##
               event=(batch['Y'][:, E-1]),  #torch.tensor
                device=device) ##
            train_loss =  loss1
            train_loss.backward()
            optimizer.step()

    # Predicting train
    rtrain_reader1 = rtrain_pred
    y_pred_list0_1 = []

    with torch.no_grad():
        model.eval()
        for batch_train in rtrain_reader1:
            cur_pred1 = model(batch_train) #, y_test_pred2 
            y_pred_list0_1.append(cur_pred1.cpu().numpy())
    y_pred_list0_1 = [a.squeeze().tolist() for a in y_pred_list0_1]
    y_pred_list0_1 = sum(y_pred_list0_1, [])

    # Predicting test
    with torch.no_grad():
        model.train() 
        for _ in range(1):   
            y_pred_list_1 = []
            for batch_test in test_loader:
                cur_out = model(batch_test)   #, y_test_pred2
                y_pred_list_1.append(cur_out.cpu().numpy())
                y_pred_list_1 = [a.squeeze().tolist() for a in y_pred_list_1]
                y_pred_list_1 = sum(y_pred_list_1, [])

    train_c = c_index(np.sum(batch_train['T'].detach().cpu().numpy(), axis = 1),  
                                                      np.exp(y_pred_list0_1), 
                                                      batch_train['Y'][:, E-1].cpu())
    test_c = c_index(np.sum(batch_test['T'].detach().cpu().numpy(), axis = 1), 
                                                     np.exp(y_pred_list_1), 
                                                     batch_test['Y'][:, E-1].cpu())
    return train_c, test_c
     #event 1

def cox_regression(data, event=None, duration=None, penalizer=0.01):
    train, test = data
    cph = lifelines.CoxPHFitter(penalizer=penalizer)
    cph.fit(train, duration_col=duration, event_col=event)
    
    train_c = lifelines.utils.concordance_index(train[duration], -np.exp(cph.predict_partial_hazard(train)), train[event]) ##
    test_c = lifelines.utils.concordance_index(test[duration], -np.exp(cph.predict_partial_hazard(test)), test[event]) ##
    return train_c, test_c

def deepsurv(data, feature, event, duration, device, batch_size = 256, num_nodes = [32,32], out_features = 1, batch_norm = True, drop_out = 0.1, output_bias= True):
    train, test = data
    x_train = train[feature]
    x_test = test[feature]
    
    y_train = train[[event, duration]]
    y_train_inp = (y_train.iloc[:,1].to_numpy(dtype='f'), y_train.iloc[:,0].to_numpy(dtype='f'))
    
    time_train, event_train = np.array(train[duration]), np.array(train[event])
    time_test, event_test = np.array(test[duration]), np.array(test[event])
    
    in_features = x_train.shape[1] 

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, drop_out, output_bias = output_bias)
    model_ds = CoxPH(net, tt.optim.Adam, device)
    
    model_ds.optimizer.set_lr(0.0001)
    epochs = 512
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = True
    log = model_ds.fit(x_train.to_numpy(dtype='f'), y_train_inp, batch_size, epochs, callbacks, verbose)

    model_ds.compute_baseline_hazards()
    
    surv1 = model_ds.predict_surv_df(x_train.to_numpy(dtype='f'))
    ev1 = EvalSurv(surv1, time_train, event_train, censor_surv='km')
    train_c = ev1.concordance_td()
    
    surv2 = model_ds.predict_surv_df(x_test.to_numpy(dtype='f'))
    ev2 = EvalSurv(surv2, time_test, event_test, censor_surv='km')
    test_c = ev2.concordance_td()
    return train_c, test_c

def evaluation_mtl_deephit(model, train_loader, train_loader_pred, test_loader, task_nums, n_epochs, learning_rate = 1e-3, loss_func=loss_func, concordance_index=concordance_index, device=device2):
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
	# Training
	for e in range(1, n_epochs+1):
		model.train()
		for X_train_batch, lifetime_batch, event_batches in train_loader:
			optimizer.zero_grad()
			yhat = model(X_train_batch)
			train_loss = []
			for i in range(task_nums):
				loss = loss_func(pred = yhat[i], 
                     lifetime=lifetime_batch, 
                     event=event_batches[:,i], device=device) #
				train_loss.append(loss)
			train_loss = sum(train_loss)/len(train_loss)
			train_loss.backward()
			optimizer.step()
	# Predicting train
	for i in range(task_nums):
		globals()['y_pred_list0_'+str(i+1)] = []
	with torch.no_grad():
		model.eval()
		for X_batch, lifetime_pred_train, event_pred_train in train_loader_pred:
			X_batch = X_batch.to(device)
			y_test_pred = model(X_batch)
			for i in range(task_nums):
				globals()['y_pred_list0_'+str(i+1)].append(y_test_pred[i])
	for i in range(task_nums):
		globals()['y_pred_list0_'+str(i+1)] = [a.squeeze().tolist() for a in globals()['y_pred_list0_'+str(i+1)]]
		globals()['y_pred_list0_'+str(i+1)] = sum(globals()['y_pred_list0_'+str(i+1)], [])
	# Predicting test
	with torch.no_grad():
		model.train()
		for _ in range(20):
			for i in range(task_nums):
				globals()['y_pred_list_'+str(i+1)] = []
			for X_batch, lifetime_pred_test, event_pred_test in test_loader:
				y_test_pred = model(X_batch)
				for i in range(task_nums):
					globals()['y_pred_list_'+str(i+1)].append(y_test_pred[i].cpu().numpy())
					globals()['y_pred_list_'+str(i+1)] = [a.squeeze().tolist() for a in globals()['y_pred_list_'+str(i+1)]]
					globals()['y_pred_list_'+str(i+1)] = sum(globals()['y_pred_list_'+str(i+1)], [])
	results = {}
	for i in range(task_nums):
		train_c = concordance_index(lifetime_pred_train.cpu(), np.exp(globals()['y_pred_list0_'+str(i+1)]), event_pred_train[:,i].cpu())
		test_c = concordance_index(lifetime_pred_test.cpu(), np.exp(globals()['y_pred_list_'+str(i+1)]), event_pred_test[:,i].cpu())
		results[i] = [train_c, test_c]
	return results

def evaluation_mtl_lstm(model, train_loader, rtrain_pred, test_loader, task_nums, learning_rate= 1e-3, n_epochs=2, loss_func=loss_func, c_index=concordance_index, device=device2):
    torch.manual_seed(1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-8)

    # Training
    for i in range(1, n_epochs+1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            yhat = model(batch, device) #0826
            train_loss = []
            for i in range(task_nums):    
                loss = loss_func(pred = yhat[i], 
                                 lifetime=torch.tensor(np.sum(batch['T'].detach().cpu().numpy(), axis = 1)).to(device), #0826 
                                 event=torch.tensor(batch['Y'][:, i]), device=device) # 0826
                train_loss.append(loss)
            train_loss =  sum(train_loss)/len(train_loss)
            train_loss.backward()
            optimizer.step()

    # Predicting train
    for i in range(task_nums):
        globals()['y_pred_list0_'+str(i+1)] = []
        # y_pred_list0_1 = []

    with torch.no_grad():
        model.eval()
        for batch_train in rtrain_pred:
            y_train_pred = model(batch_train, device) #0826
            for i in range(task_nums):
                globals()['y_pred_list0_'+str(i+1)].append(y_train_pred[i].cpu().numpy()) # 0826
            # y_pred_list0_1.append(cur_pred1[0].cpu().numpy())
    for i in range(task_nums):
        globals()['y_pred_list0_'+str(i+1)] = [a.squeeze().tolist() for a in globals()['y_pred_list0_'+str(i+1)]]
        globals()['y_pred_list0_'+str(i+1)] = sum(globals()['y_pred_list0_'+str(i+1)], [])
    # y_pred_list0_1 = [a.squeeze().tolist() for a in y_pred_list0_1]
    # y_pred_list0_1 = sum(y_pred_list0_1, [])

    # Predicting test
    with torch.no_grad():
        model.train() 
        for _ in range(1):
            for i in range(task_nums):
                globals()['y_pred_list_'+str(i+1)] = []
            # y_pred_list_1 = []
            for batch_test in test_loader:
                y_test_pred = model(batch_test, device) #0826
                for i in range(task_nums):
                    globals()['y_pred_list_'+str(i+1)].append(y_test_pred[i].cpu().numpy())
                    globals()['y_pred_list_'+str(i+1)] = [a.squeeze().tolist() for a in globals()['y_pred_list_'+str(i+1)]]
                    globals()['y_pred_list_'+str(i+1)] = sum(globals()['y_pred_list_'+str(i+1)], [])
                # y_pred_list_1.append(cur_out[0].cpu().numpy())
                # y_pred_list_1 = [a.squeeze().tolist() for a in y_pred_list_1]
                # y_pred_list_1 = sum(y_pred_list_1, [])
      
    results = {}            
    for i in range(task_nums):
        train_c = c_index(np.sum(batch_train['T'].detach().cpu().numpy(), axis = 1),  #0826
                          np.exp(globals()['y_pred_list0_'+str(i+1)]), 
                          batch_train['Y'][:, i].cpu()) #0826
        test_c = c_index(np.sum(batch_test['T'].detach().cpu().numpy(), axis = 1),  #0826
                         np.exp(globals()['y_pred_list_'+str(i+1)]), 
                         batch_test['Y'][:, i].cpu()) #0826
        results[i] = [train_c, test_c]
    return results
    
class LightninglstmClassifier(pl.LightningModule):

    def __init__(self, 
                input_size = 12, #n_feature, 
                layer_hidden_sizes = [3,3,5],
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

    def surv_loss(self, pred, lifetime, event): #, device
        return loss_func(pred, lifetime, event) #, device

    def training_step(self, train_batch, batch_idx):
        # x, y = train_batch
        E = self.E
        _, yhat = self.forward(train_batch) #? self.model?
        loss = self.surv_loss(pred = yhat, 
                              lifetime=torch.tensor(np.sum(train_batch['T'].detach().numpy(), axis = 1)),
                               event=torch.tensor(train_batch['Y'][:, E-1]),
                               ) #device=device2
        self.log('train_loss', loss)
        
        return {'loss': loss, 'pred': yhat, 'T': train_batch['T'], 'event': train_batch['Y'][:, E-1]}


    def validation_step(self, val_batch, batch_idx):
        E = self.E
        # x, y = val_batch
        _, yhat = self.forward(val_batch)
        loss = self.surv_loss(pred = yhat, 
                              lifetime=torch.tensor(np.sum(val_batch['T'].detach().numpy(), axis = 1)),
                               event=torch.tensor(val_batch['Y'][:, E-1]),
                               ) #device = device2

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
    
class Lightningmtl_lstm(pl.LightningModule):

    def __init__(self, 
                 in_features, 
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

    def forward(self, input_data): #0826 , device
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
                               (self.shared_layers[0]+1) * self.in_features)
        for i in range(len(X)):    
            x_series[i][:len(X[i]), :] = X[i] 
        X = x_series #0826 .to(device)
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

    def surv_loss(self, pred, lifetime, event): ## , device
        return loss_func(pred, lifetime, event) ## , device

    def training_step(self, train_batch, batch_idx): #, device
        task_nums = train_batch['Y'].shape[1] #[0]
        yhat = self.forward(train_batch) # , device
        train_loss = []
        for i in range(task_nums):
            loss = self.surv_loss(pred = yhat[i], 
                                lifetime=torch.tensor(np.sum(train_batch['T'].detach().numpy(), axis = 1)), #to device?
                                event=torch.tensor(train_batch['Y'][:, i]),
                                ) #device=device2
            train_loss.append(loss)
        train_loss =  sum(train_loss)/len(train_loss)
        self.log('train_loss', train_loss) #list value cannot be logged detach()
        
        return {'loss': train_loss, 'pred': yhat, 'T': train_batch['T'], 'event': train_batch['Y'][:, i]} #.detach() list has no attribute detach()


    def validation_step(self, val_batch, batch_idx): #, device
        task_nums = val_batch['Y'].shape[1] #[0]
        yhat = self.forward(val_batch) #, device
        val_loss = []
        for i in range(task_nums):
            loss = self.surv_loss(pred = yhat[i], 
                                lifetime=torch.tensor(np.sum(val_batch['T'].detach().numpy(), axis = 1)), #to device?
                                event=torch.tensor(val_batch['Y'][:, i]),
                                )#device=device2
            val_loss.append(loss)
        val_loss =  sum(val_loss)/len(val_loss)
        self.log('val_loss', val_loss) # detach()

    def test_step(self, test_batch, batch_idx): #, device
        task_nums = test_batch['Y'].shape[1] #[0]
        yhat = self.forward(test_batch) #, device
        accs = {}
        for i in range(task_nums):
            acc = concordance_index(np.sum(test_batch['T'].detach().numpy(), axis = 1),  
                                    np.exp(yhat[i]), #? 
                                    test_batch['Y'][:, i])
            accs['event_' + str(i+1)] = acc
            # # self.log(globals()['c-index_' + str(i+1)], acc) #accs
        self.log('c-index', accs) # accs Problem 0828, why acc <0.01

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
