import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
os.environ["RAY_ENABLE_METRICS_COLLECTION"] = "0"
os.environ["RAY_LOG_TO_STDERR"] = "0"

import math
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader
from geopy.distance import geodesic

from script import utility
from model import models

class Args:
    def __init__(self, dataset="data_PEMSD7"):
        self.dataset = dataset
        self.n_his = 12
        self.n_pred = 3
        self.time_intvl = 5
        self.Kt = 3
        self.stblock_num = 2
        self.act_func = 'glu'
        self.Ks = 3
        self.graph_conv_type = 'cheb_graph_conv'
        self.gso_type = 'sym_norm_lap'
        self.enable_bias = True
        self.droprate = 0.5
        self.gso = None

def load_adj(dataset_path):
    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    adj = adj.tocsc()
    n_vertex = adj.shape[0]
    return adj, n_vertex

def get_cloudlets(locations_json_path, cloudlet_experiment):
    with open(locations_json_path) as f:
        data = json.load(f)
    return data[cloudlet_experiment]["cloudlets"], data[cloudlet_experiment]["radius_km"]

def is_within_radius(lat1, lon1, lat2, lon2, radius_km):
    return geodesic((lat1, lon1), (lat2, lon2)).km <= radius_km

def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

def partition_nodes_to_cloudlets_by_range_proximity(cloudlets, radius_km, dataset_path):
    locations_data = pd.read_csv(os.path.join(dataset_path, 'locations-raw.csv'))

    cloudlet_nodes_list = [[] for _ in range(len(cloudlets))]

    for idx, sensor in locations_data.iterrows():
        sensor_loc = (sensor['Latitude'], sensor['Longitude']) 
        closest_cloudlet = None
        min_distance = float('inf')

        for name, loc in cloudlets.items():
            if is_within_radius(sensor_loc[0], sensor_loc[1], loc['lat'], loc['lon'], radius_km):
                distance = calculate_distance(sensor_loc[0], sensor_loc[1], loc['lat'], loc['lon'])
                if distance < min_distance:
                    min_distance = distance
                    closest_cloudlet = loc['id']

        if closest_cloudlet is not None:
            cloudlet_nodes_list[closest_cloudlet].append(idx)

    return cloudlet_nodes_list

def load_data(dataset_path):
    vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))
    
    data_col = vel.shape[0]
    
    # 30% initial, 40% online, 30% eval
    len_initial = int(math.floor(data_col * 0.30))
    len_online = int(math.floor(data_col * 0.40))
    len_eval = int(data_col - len_initial - len_online)
    
    train = vel[: len_initial + len_online]
    test = vel[len_initial + len_online:] 
    
    return train, test, len_initial, len_online

def data_transform(data, n_his, n_pred, device):
    if len(data) == 0:
        return torch.Tensor([]).to(device), torch.Tensor([]).to(device)
    data_values = data.values if isinstance(data, pd.DataFrame) else data
    n_vertex = data_values.shape[1]
    len_record = len(data_values)
    num = len_record - n_his - n_pred
    
    if num <= 0:
        return torch.Tensor([]).to(device), torch.Tensor([]).to(device)
        
    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, n_vertex])
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data_values[head: tail].reshape(1, n_his, n_vertex)
        y[i] = data_values[tail + n_pred - 1]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)

def get_blocks(args):
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num
    blocks = []
    blocks.append([1])
    for l in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])
    return blocks

def prepare_model(dataset_name, dataset_path, device=torch.device("cpu")):
    args = Args(dataset=dataset_name)
    adj, n_vertex = load_adj(dataset_path)
    gso = utility.calc_gso(adj, args.gso_type)
    if args.graph_conv_type == 'cheb_graph_conv':
        gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso).to(device)
    
    blocks = get_blocks(args)
    
    if args.graph_conv_type == 'cheb_graph_conv':
        model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
    else:
        model = models.STGCNGraphConv(args, blocks, n_vertex).to(device)
        
    return model

def load_flower_data(dataset_name, partition_id, num_partitions, batch_size, online_steps, locations_json_path, cloudlet_experiment, dataset_path, device=torch.device("cpu")):
    args = Args(dataset=dataset_name)
    
    cloudlets, radius_km = get_cloudlets(locations_json_path, cloudlet_experiment)
    cln_nodes_list = partition_nodes_to_cloudlets_by_range_proximity(cloudlets, radius_km, dataset_path)
    cln_nodes = cln_nodes_list[partition_id]
    
    node_map = torch.tensor(cln_nodes, dtype=torch.long).to(device)
    
    train_full, test, len_initial, len_online = load_data(dataset_path)
    
    zscore = preprocessing.StandardScaler()
    train_scaled = zscore.fit_transform(train_full)
    test_scaled = zscore.transform(test)
    
    x_train, y_train = data_transform(train_scaled, args.n_his, args.n_pred, device)
    x_test, y_test = data_transform(test_scaled, args.n_his, args.n_pred, device)
    
    end_of_initial_data_index = len_initial - (args.n_his + args.n_pred)
    data_per_step = (x_train.shape[0] - end_of_initial_data_index) // online_steps
    
    test_data = TensorDataset(x_test, y_test)
    test_iter = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    
    return x_train, y_train, end_of_initial_data_index, data_per_step, test_iter, node_map

def create_train_iter_for_online(epoch, x_train, y_train, end_of_initial_data_index, data_per_step, batch_size):
    if epoch == 0:
        inital_x_train = x_train[:end_of_initial_data_index]
        inital_y_train = y_train[:end_of_initial_data_index]
        train_data = TensorDataset(inital_x_train, inital_y_train)
        return DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    else:
        current_len = end_of_initial_data_index + (data_per_step * (epoch - 1))
        
        random_sample_size = (batch_size - 1) * data_per_step
        if current_len > random_sample_size:
            random_indices = np.random.choice(current_len, random_sample_size, replace=False)
        else:
            random_indices = np.arange(current_len)
            
        new_x_train = x_train[
            end_of_initial_data_index + (data_per_step * (epoch - 1)):
            end_of_initial_data_index + (data_per_step * (epoch))
        ]
        sampled_x_train = x_train[random_indices, :]
        new_x_train = torch.cat((sampled_x_train, new_x_train), dim=0)
        
        new_y_train = y_train[
            end_of_initial_data_index + (data_per_step * (epoch - 1)):
            end_of_initial_data_index + (data_per_step * (epoch))
        ]
        sampled_y_train = y_train[random_indices, :]
        new_y_train = torch.cat((sampled_y_train, new_y_train), dim=0)

        train_data = TensorDataset(new_x_train, new_y_train)
        return DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

def train_online(model, x_train, y_train, end_of_initial_data_index, data_per_step, node_map, epochs, lr, batch_size, online_steps, device, partition_id):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=0.001)
    
    train_iter = create_train_iter_for_online(0, x_train, y_train, end_of_initial_data_index, data_per_step, batch_size)
    model.train()
    for _ in range(epochs): 
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x).view(len(x), -1) 
            y_pred = y_pred[:, node_map]
            y_mapped = y[:, node_map]
            l = loss_fn(y_pred, y_mapped)
            l.backward()
            optimizer.step()
            
    l_sum, n = 0.0, 0
    for online_step in range(1, online_steps + 1):
        train_iter = create_train_iter_for_online(online_step, x_train, y_train, end_of_initial_data_index, data_per_step, batch_size)
        model.train()
        step_l_sum, step_n = 0.0, 0
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x).view(len(x), -1)
            y_pred = y_pred[:, node_map]
            y_mapped = y[:, node_map]
            l = loss_fn(y_pred, y_mapped)
            l.backward()
            optimizer.step()
            
            step_l_sum += l.item() * y.shape[0]
            step_n += y.shape[0]
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
            
    return l_sum / n if n > 0 else 0.0

@torch.no_grad()
def test(model, val_iter, node_map):
    model.eval()
    loss_fn = nn.MSELoss()
    
    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x).view(len(x), -1)
        y_pred = y_pred[:, node_map]
        y_mapped = y[:, node_map]
        l = loss_fn(y_pred, y_mapped)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
        
    return l_sum / n if n > 0 else 0.0
