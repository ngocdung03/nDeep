import torch
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing
import pandas as pd
import copy
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DatasetReader(torch.utils.data.Dataset):
    def __init__(self, data_dict, reverse = False, data_type = 'distribute', maxlength_seq = 192, device=device):
        super().__init__() #?  self,data
        self.data_type = data_type
        self.feat_info = data_dict['x']
        if data_type == 'aggregation':
            self.time = data_dict['t']
        self.label_list = data_dict['y']
        self.seq_len = 192
        # self.reverse = reverse
        self.time = data_dict['time']
    def __getitem__(self, index):
        if self.data_type == 'distribute':
            s_data = copy.deepcopy(self.feat_info[index].to_numpy())
            covariates = s_data[:self.seq_len]
            time = copy.deepcopy(self.time[index].to_numpy())
        else:
            covariates = self.feat_info[index]
            time = self.time[index]
        l, w = np.shape(covariates)
        time[0] = 0.
        if len(time) >0:
           time[1:] = time[1:] - time[:-1]
           time[0] = 0.
           if time[-1] < 0:
               time[-1] = self.time[index].to_numpy()[-1]
        x_time = np.zeros(self.seq_len)
        x_time[:l] = time
        x_series = np.zeros([self.seq_len, w])
        x_series[:l, :] = covariates
        x_mask = np.zeros(self.seq_len)
        x_mask[:l] = 1.
        x_mask_cur = np.zeros(self.seq_len)
        x_mask_cur[l-1] = 1.
        label = self.label_list[index]
        return {'X': torch.from_numpy(np.array(x_series)).to(device), 
                'M': torch.from_numpy(np.array(x_mask)).to(device), 
                'cur_M': torch.from_numpy(np.array(x_mask_cur)).to(device), 
                'Y': torch.from_numpy(np.array(label).astype(np.float64)).to(device), 
                'T': torch.from_numpy(np.array(x_time)).to(device)}  #.to(device)
    def __len__(self):
        return len(self.feat_info)


def _multiply(args):
    df, que = args
    que.put([pd.concat([pd.DataFrame(x[-1]).T]*int(x[-1]['time']), ignore_index=True) if x[-1]['time'] == int(x[-1]['time']) else pd.concat([pd.DataFrame(x[-1]).T]*(int(x[-1]['time']) + 1), ignore_index=True) for x in df.iterrows()])
    return

def get_loader(df, feature=None, event=None, duration=None, batch_size=None,chunk=1000, num_workers = 1, device=device):
    # df.columns = ['time'] + list(df.columns)[1:] #?
    if type(event) == list:
        label = [duration] + event
    else:
        label = [duration , event]
    df = df[label + feature]
    size = int(len(df) / 1000) + 1
    n_process = int(size / 10) * 2
    if n_process < 2:
        n_process = 2
    
    split_df = [df.iloc[x*chunk: (x+1)*chunk, :] for x in range(size)] #split data if n>1000
    que = multiprocessing.Manager().Queue()
    pool = multiprocessing.Pool(n_process)
    pool.map(_multiply, [(x, que) for x in split_df])
    pool.close()
    pool.join()
    
    input_data = [que.get() for x in range(que.qsize())] #make ldata
    rdata = {
        'x' : [x[feature] for xx in input_data for x in xx],
        'y' : [x.loc[0, label[1:]] for xx in input_data for x in xx],  # x.iloc[0, [1]]
        'time': [x[duration] for xx in input_data for x in xx],
        'l': [len(x) if x[duration].iloc[-1] != 0 else 0 for xx in input_data for x in xx],
        'label_n' : len(label) - 1
    }
    if batch_size is None:
        return DataLoader(DatasetReader(rdata), batch_size=len(rdata['y']), drop_last=True, shuffle=True, num_workers=num_workers)
    return DataLoader(DatasetReader(rdata), batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers), DataLoader(DatasetReader(rdata), batch_size=len(rdata['y']), drop_last=True, shuffle=True, num_workers=num_workers)

def get_loader2(raw, 
             outcome=None,
             covariates = None,
             age_var = None,
             batch_size = None,
                device=device,
                num_workers = 0
             ):
    input_data = []
    for i in raw.index:
        time = raw.loc[i, 'time']
        time_round = int(time)
        age = int(raw.loc[i, age_var]) if age_var else None 
        dat = pd.DataFrame(raw.loc[i,:]).transpose()  #should not use iloc, otherwise indexing by location
        if time <= 1:
                dat['time'] = dat['time']
            #  dat[age_var] = age if age else dat[age_var]
        elif time == time_round:
                dat = dat.append(pd.DataFrame([raw.loc[i,:]]*
                                        (time_round - 1)), ignore_index=True)
                dat['time'] = [t for t in range(1, time_round+1)]
            #  dat[age_var] = [a for a in range(age+1 - len(dat), age+1)] if age else dat[age_var]
        elif time % time_round !=0:
                dat = dat.append(pd.DataFrame([raw.loc[i,:]]*
                                        time_round), ignore_index=True)
                dat['time'] = [t for t in range(1, time_round+1)] + [time%time_round]
            #  dat[age_var] = [a for a in range(age+1 - (len(dat)-1), age+1)] + [age+time%time_round] if age else dat[age_var]
        input_data.append(dat)
    
    if covariates == None:
            covariates = list(set(input_data[0].columns) - set(['time', 'label', 'id']))

    label_raw = raw[[outcome]] if type(outcome) != list else raw[outcome]
    label_raw = label_raw.to_numpy().astype(np.float32) if type(label_raw).__module__ != 'numpy' else label_raw    
    
    rdata = {'x': [x[covariates] for x in input_data],  #x_data, exclude id .iloc[:,:-1] 
                            'y': label_raw ,
                            'time': [x['time'] for x in input_data],      
                            'l': [len(x) if x['time'].iloc[-1]!=0 else 0 for x in input_data],   
                            # 'label_n': label_raw.shape[1] - 1
                            } 
        
    if batch_size is None:
        return DataLoader(DatasetReader(rdata, device = device), batch_size=len(rdata['y']), drop_last=True, shuffle=True, num_workers=num_workers)
    return DataLoader(DatasetReader(rdata, device = device), batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers), DataLoader(DatasetReader(rdata, device= device), batch_size=len(rdata['y']), drop_last=True, shuffle=True,num_workers=num_workers)

def get_loader_deephit(df, feature, events, duration, batch_size = None, device = device):
    covs = df[feature]
    label = df[[duration] + events]
    covs = covs.to_numpy().astype(np.float32) if type(covs).__module__ != 'numpy' else covs
    label = label.to_numpy().astype(np.float32) if type(label).__module__ != 'numpy' else label
    torch.manual_seed(1)
    tensor_dataset = TensorDataset(torch.from_numpy(covs).float().to(device), torch.from_numpy(label[:,0]).float().to(device), torch.from_numpy(label[:,1:]).float().to(device))
    if batch_size is None:
        return DataLoader(dataset=tensor_dataset, batch_size = len(df), shuffle=True, drop_last=True)
    return DataLoader(dataset=tensor_dataset, batch_size = batch_size, shuffle=True, drop_last=True), DataLoader(dataset=tensor_dataset, batch_size = len(df), shuffle=True, drop_last=True)

# if __name__ == '__main__':
#     raw = pd.read_csv('dataset/test.csv')
#     get_loader2(raw, outcome=None)
