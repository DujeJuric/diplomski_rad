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
        self.n_pred = 12
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

_ADJ_CACHE = {}
_PARTITION_CACHE = {}
_DATA_CACHE = {}

def load_adj(dataset_path):
    if dataset_path in _ADJ_CACHE:
        adj, n_vertex = _ADJ_CACHE[dataset_path]
        return adj.copy(), n_vertex

    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    adj = adj.tocsc()
    n_vertex = adj.shape[0]
    
    _ADJ_CACHE[dataset_path] = (adj, n_vertex)
    return adj.copy(), n_vertex

def get_cloudlets(locations_json_path, cloudlet_experiment):
    with open(locations_json_path) as f:
        data = json.load(f)
    return data[cloudlet_experiment]["cloudlets"], data[cloudlet_experiment]["radius_km"]

def is_within_radius(lat1, lon1, lat2, lon2, radius_km):
    return geodesic((lat1, lon1), (lat2, lon2)).km <= radius_km

def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

def partition_nodes_to_cloudlets_by_range_proximity(cloudlets, radius_km, dataset_path):
    cache_key = (str(cloudlets), radius_km, dataset_path)
    if cache_key in _PARTITION_CACHE:
        return [list(lst) for lst in _PARTITION_CACHE[cache_key]]

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

    _PARTITION_CACHE[cache_key] = [list(lst) for lst in cloudlet_nodes_list]
    return cloudlet_nodes_list

def load_data(dataset_path):
    if dataset_path in _DATA_CACHE:
        train, test, len_initial, len_online = _DATA_CACHE[dataset_path]
        return train.copy(), test.copy(), len_initial, len_online

    vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))
    
    len_initial = 3796
    len_online = 5080
    
    train = vel[: len_initial + len_online]
    test = vel[len_initial + len_online:] 
    
    _DATA_CACHE[dataset_path] = (train, test, len_initial, len_online)
    return train.copy(), test.copy(), len_initial, len_online

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
    
    train, test, len_initial, len_online = load_data(dataset_path)
    
    zscore = preprocessing.StandardScaler()
    train_scaled = zscore.fit_transform(train)
    test_scaled = zscore.transform(test)
    
    x_train, y_train = data_transform(train_scaled, args.n_his, args.n_pred, device)
    x_test, y_test = data_transform(test_scaled, args.n_his, args.n_pred, device)
    
    end_of_initial_data_index = len_initial - (args.n_his + args.n_pred)
    data_per_step = (x_train.shape[0] - end_of_initial_data_index) // online_steps
    
    test_data = TensorDataset(x_test, y_test)
    test_iter = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    
    return x_train, y_train, end_of_initial_data_index, data_per_step, test_iter, node_map, zscore

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

def save_metrics_plot(partition_id, history):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    graphs_dir = os.path.join(current_dir, "stgcn_graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    try:
        import json
        with open(os.path.join(graphs_dir, f"stgcn_client_{partition_id}_metrics.json"), "w") as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print("fail")

    try:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Client {partition_id}")
        
        axs[0, 0].plot(history["step"], history["loss"], marker='o', color='blue')
        axs[0, 0].set_title("Loss")
        axs[0, 0].set_xlabel("Online Step")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].grid(True)
        
        axs[0, 1].plot(history["step"], history["mae"], marker='s', color='orange')
        axs[0, 1].set_title("MAE")
        axs[0, 1].set_xlabel("Online Step")
        axs[0, 1].set_ylabel("MAE")
        axs[0, 1].grid(True)
        
        axs[1, 0].plot(history["step"], history["rmse"], marker='^', color='green')
        axs[1, 0].set_title("RMSE")
        axs[1, 0].set_xlabel("Online Step")
        axs[1, 0].set_ylabel("RMSE")
        axs[1, 0].grid(True)
        
        axs[1, 1].plot(history["step"], history["mape"], marker='d', color='red')
        axs[1, 1].set_title("MAPE")
        axs[1, 1].set_xlabel("Online Step")
        axs[1, 1].set_ylabel("MAPE")
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, f"stgcn_client_{partition_id}_metrics.png"), dpi=150)
        plt.close()
    except ImportError:
        print("error")
       

def train_online(model, x_train, y_train, end_of_initial_data_index, data_per_step, node_map, epochs, lr, batch_size, online_steps, device, partition_id, val_iter=None, scaler=None):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=0.001)
    
    history = {"step": [], "loss": [], "mae": [], "rmse": [], "mape": []}

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
            
    if val_iter is not None and scaler is not None:
        val_loss, mae, rmse, mape = test(model, val_iter, node_map, scaler)
        print(f"Client {partition_id} | Initial Step | Loss: {val_loss:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.4f}%", flush=True)
        history["step"].append(0)
        history["loss"].append(val_loss)
        history["mae"].append(mae)
        history["rmse"].append(rmse)
        history["mape"].append(mape)
        save_metrics_plot(partition_id, history)

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
            
        if val_iter is not None and scaler is not None:
            val_loss, mae, rmse, mape = test(model, val_iter, node_map, scaler)
            print(f"Client {partition_id} | Online Step {online_step:2d} | Loss: {val_loss:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.4f}%", flush=True)
            history["step"].append(online_step)
            history["loss"].append(val_loss)
            history["mae"].append(mae)
            history["rmse"].append(rmse)
            history["mape"].append(mape)
            save_metrics_plot(partition_id, history)

    return l_sum / n if n > 0 else 0.0

@torch.no_grad()
def test(model, val_iter, node_map, scaler):
    model.eval()
    loss_fn = nn.MSELoss()
    
    l_sum, n = 0.0, 0
    mae_sum = 0.0
    mse_sum = 0.0
    mape_sum = 0.0
    total_elements = 0
    
    for x, y in val_iter:
        y_pred = model(x).view(len(x), -1)
        y_pred_mapped = y_pred[:, node_map]
        y_mapped = y[:, node_map]
        l = loss_fn(y_pred_mapped, y_mapped)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
      
        y_unscaled = scaler.inverse_transform(y.cpu().numpy())
        y_pred_unscaled = scaler.inverse_transform(y_pred.cpu().numpy())
    
        y_unscaled_mapped = y_unscaled[:, node_map.cpu().numpy()]
        y_pred_unscaled_mapped = y_pred_unscaled[:, node_map.cpu().numpy()]
        
        diff = np.abs(y_pred_unscaled_mapped - y_unscaled_mapped)
        
        mae_sum += np.sum(diff)
        mse_sum += np.sum(diff ** 2)
    
        mape_sum += np.sum(diff / np.where(y_unscaled_mapped == 0, 1e-5, y_unscaled_mapped))
        total_elements += diff.size
        
    avg_loss = l_sum / n if n > 0 else 0.0
    avg_mae = mae_sum / total_elements if total_elements > 0 else 0.0
    avg_rmse = np.sqrt(mse_sum / total_elements) if total_elements > 0 else 0.0
    avg_mape = (mape_sum / total_elements) * 100 if total_elements > 0 else 0.0
    
    return avg_loss, avg_mae, avg_rmse, avg_mape
