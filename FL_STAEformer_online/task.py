import os
import sys
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

from model.STAEformer import STAEformer

def load_adj(dataset_path):
    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
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
    
    # 30% initial, 40% online, 30% test
    initial_rate = 0.3
    online_rate = 0.4
    
    len_initial = int(math.floor(data_col * initial_rate))
    len_online = int(math.floor(data_col * online_rate))
    
    train_full = vel[: len_initial + len_online]
    test = vel[len_initial + len_online:]
    
    return train_full, test, len_initial, len_online

def data_transform(data, in_steps, out_steps, device):
    if len(data) == 0:
        return torch.Tensor([]).to(device), torch.Tensor([]).to(device)
    data_values = data.values if isinstance(data, pd.DataFrame) else data
    n_vertex = data_values.shape[1]
    len_record = len(data_values)
    num = len_record - in_steps - out_steps
    
    if num <= 0:
        return torch.Tensor([]).to(device), torch.Tensor([]).to(device)
        
    x = np.zeros([num, in_steps, n_vertex, 1])
    y = np.zeros([num, out_steps, n_vertex, 1])
    
    for i in range(num):
        x[i, :, :, 0] = data_values[i: i+in_steps]
        y[i, :, :, 0] = data_values[i+in_steps: i+in_steps+out_steps]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)

def prepare_model(dataset_path, device=torch.device("cpu")):
    _, n_vertex = load_adj(dataset_path)
    
    model_args = {
        "num_nodes": n_vertex,
        "in_steps": 12,
        "out_steps": 12,
        "steps_per_day": 288,
        "input_dim": 1, 
        "output_dim": 1,
        "input_embedding_dim": 8,
        "tod_embedding_dim": 0,
        "dow_embedding_dim": 0,
        "spatial_embedding_dim": 0,
        "adaptive_embedding_dim": 16,
        "feed_forward_dim": 64,
        "num_heads": 2,
        "num_layers": 1,
        "dropout": 0.1,
    }
    
    model = STAEformer(**model_args).to(device)
    return model

def load_flower_data(partition_id, num_partitions, batch_size, online_steps, locations_json_path, cloudlet_experiment, dataset_path, device=torch.device("cpu")):

    cloudlets, radius_km = get_cloudlets(locations_json_path, cloudlet_experiment)
    cln_nodes_list = partition_nodes_to_cloudlets_by_range_proximity(cloudlets, radius_km, dataset_path)
    cln_nodes = cln_nodes_list[partition_id]
    
    node_map = torch.tensor(cln_nodes, dtype=torch.long).to(device)

    train_full, test, len_initial, len_online = load_data(dataset_path)
    
    zscore = preprocessing.StandardScaler()
    train_scaled = zscore.fit_transform(train_full)
    test_scaled = zscore.transform(test)
    
    x_train, y_train = data_transform(train_scaled, 12, 12, device)
    x_test, y_test = data_transform(test_scaled, 12, 12, device)
    
    end_of_initial_data_index = len_initial - (12 + 12)
    data_per_step = (x_train.shape[0] - end_of_initial_data_index) // online_steps
    
    test_data = TensorDataset(x_test, y_test)
    test_iter = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    
    return x_train, y_train, end_of_initial_data_index, data_per_step, test_iter, node_map

def create_train_iter_for_online(epoch, x_train, y_train, end_of_initial_data_index, data_per_step, batch_size):
    if epoch == 0:
        new_x_train = x_train[0 : end_of_initial_data_index]
        new_y_train = y_train[0 : end_of_initial_data_index]
        train_data = TensorDataset(new_x_train, new_y_train)
        return DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    else:
        current_len = end_of_initial_data_index + (data_per_step * (epoch - 1))
        random_sample_size = (batch_size - 1) * data_per_step
        
        if current_len > random_sample_size:
            random_indices = np.random.choice(current_len, random_sample_size, replace=False)
        else:
            random_indices = np.arange(current_len)
            
        sampled_x_train = x_train[random_indices, :]
        sampled_y_train = y_train[random_indices, :]
        
        new_x_train = x_train[
            end_of_initial_data_index + (data_per_step * (epoch - 1)):
            end_of_initial_data_index + (data_per_step * (epoch))
        ]
        new_y_train = y_train[
            end_of_initial_data_index + (data_per_step * (epoch - 1)):
            end_of_initial_data_index + (data_per_step * (epoch))
        ]
        
        new_x_train = torch.cat((sampled_x_train, new_x_train), dim=0)
        new_y_train = torch.cat((sampled_y_train, new_y_train), dim=0)

        train_data = TensorDataset(new_x_train, new_y_train)
        return DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

def train_online(model, x_train, y_train, end_of_initial_data_index, data_per_step, node_map, epochs, lr, batch_size, online_steps, device, partition_id):
    loss_fn = nn.HuberLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.001)
    
    train_iter = create_train_iter_for_online(0, x_train, y_train, end_of_initial_data_index, data_per_step, batch_size)
    model.train()
    for _ in range(epochs): 
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x)
            y_pred_masked = y_pred[:, :, node_map, :]
            y_mapped = y[:, :, node_map, :]
            l = loss_fn(y_pred_masked, y_mapped)
            l.backward()
            optimizer.step()
            
    l_sum, n = 0.0, 0
    for online_step in range(1, online_steps + 1):
        train_iter = create_train_iter_for_online(online_step, x_train, y_train, end_of_initial_data_index, data_per_step, batch_size)
        model.train()
        step_l_sum, step_n = 0.0, 0
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x)
            y_pred_masked = y_pred[:, :, node_map, :]
            y_mapped = y[:, :, node_map, :]
            l = loss_fn(y_pred_masked, y_mapped)
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
    loss_fn = nn.HuberLoss()
    
    l_sum, n = 0.0, 0
    for x, y in val_iter:
        y_pred = model(x)
        y_pred_masked = y_pred[:, :, node_map, :]
        y_mapped = y[:, :, node_map, :]
        l = loss_fn(y_pred_masked, y_mapped)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
        
    return l_sum / n if n > 0 else 0.0
