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

_ADJ_CACHE = {}
_PARTITION_CACHE = {}
_DATA_CACHE = {}

def load_adj(dataset_path):
    if dataset_path in _ADJ_CACHE:
        adj, n_vertex = _ADJ_CACHE[dataset_path]
        return adj.copy(), n_vertex

    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
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

def data_transform(data, in_steps, out_steps, device, start_idx=0):
    if len(data) == 0:
        return torch.Tensor([]).to(device), torch.Tensor([]).to(device)
    data_values = data.values if isinstance(data, pd.DataFrame) else data
    n_vertex = data_values.shape[1]
    len_record = len(data_values)
    num = len_record - in_steps - out_steps
    
    if num <= 0:
        return torch.Tensor([]).to(device), torch.Tensor([]).to(device)
        
    x = np.zeros([num, in_steps, n_vertex, 3])
    y = np.zeros([num, out_steps, n_vertex, 1])
    
    for i in range(num):
        x[i, :, :, 0] = data_values[i: i+in_steps]
        
        abs_steps = np.arange(start_idx + i, start_idx + i + in_steps)
        tod = (abs_steps % 288) / 288.0
        dow = (abs_steps // 288) % 7
        
        x[i, :, :, 1] = np.tile(tod[:, np.newaxis], (1, n_vertex))
        x[i, :, :, 2] = np.tile(dow[:, np.newaxis], (1, n_vertex))
        
        y[i, :, :, 0] = data_values[i+in_steps: i+in_steps+out_steps]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)

def prepare_model(dataset_name, dataset_path, device=torch.device("cpu")):
    _, n_vertex = load_adj(dataset_path)
    
    model_args = {
        "num_nodes": n_vertex,
        "in_steps": 12,
        "out_steps": 12,
        "steps_per_day": 288,
        "input_dim": 3, 
        "output_dim": 1,
        #max 24, 8
        "input_embedding_dim": 8,
        "tod_embedding_dim": 8,
        "dow_embedding_dim": 8,
        "spatial_embedding_dim": 0,
        #max 80, 16
        "adaptive_embedding_dim": 16,
        #max 256, 64
        "feed_forward_dim": 64,
        #max 4, 2
        "num_heads": 2,
        #max 3, 1
        "num_layers": 1,
        "dropout": 0.1,
    }
    
    model = STAEformer(**model_args).to(device)
    return model

def load_flower_data(dataset_name, partition_id, num_partitions, batch_size, online_steps, locations_json_path, cloudlet_experiment, dataset_path, device=torch.device("cpu")):

    cloudlets, radius_km = get_cloudlets(locations_json_path, cloudlet_experiment)
    cln_nodes_list = partition_nodes_to_cloudlets_by_range_proximity(cloudlets, radius_km, dataset_path)
    cln_nodes = cln_nodes_list[partition_id]
    
    node_map = torch.tensor(cln_nodes, dtype=torch.long).to(device)

    train, test, len_initial, len_online = load_data(dataset_path)
    
    zscore = preprocessing.StandardScaler()
    train_scaled = zscore.fit_transform(train)
    test_scaled = zscore.transform(test)
    
    in_steps = 12
    out_steps = 12
    x_train, y_train = data_transform(train_scaled, in_steps, out_steps, device, start_idx=0)
    x_test, y_test = data_transform(test_scaled, in_steps, out_steps, device, start_idx=len_initial + len_online)
    
    end_of_initial_data_index = len_initial - (in_steps + out_steps)
    data_per_step = (x_train.shape[0] - end_of_initial_data_index) // online_steps
    
    test_data = TensorDataset(x_test, y_test)
    test_iter = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    
    return x_train, y_train, end_of_initial_data_index, data_per_step, test_iter, node_map, zscore

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

def save_metrics_plot(partition_id, history):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    graphs_dir = os.path.join(current_dir, "staeformer_graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    try:
        import json
        with open(os.path.join(graphs_dir, f"staeformer_client_{partition_id}_metrics.json"), "w") as f:
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
        plt.savefig(os.path.join(graphs_dir, f"staeformer_client_{partition_id}_metrics.png"), dpi=150)
        plt.close()
    except ImportError:
        print("error")

def train_online(model, x_train, y_train, end_of_initial_data_index, data_per_step, node_map, epochs, lr, batch_size, online_steps, device, partition_id, val_iter=None, scaler=None):
    loss_fn = nn.HuberLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.001)
    
    history = {"step": [], "loss": [], "mae": [], "rmse": [], "mape": []}

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
    loss_fn = nn.HuberLoss()
    
    l_sum, n = 0.0, 0
    mae_sum = 0.0
    mse_sum = 0.0
    mape_sum = 0.0
    total_elements = 0
    
    for x, y in val_iter:
        y_pred = model(x)
        y_pred_masked = y_pred[:, :, node_map, :]
        y_mapped = y[:, :, node_map, :]
        l = loss_fn(y_pred_masked, y_mapped)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
        
        n_vertex = y.shape[2]
        
        y_2d = y.cpu().numpy()[..., 0].reshape(-1, n_vertex)
        y_pred_2d = y_pred.cpu().numpy()[..., 0].reshape(-1, n_vertex)
        
        y_unscaled_2d = scaler.inverse_transform(y_2d)
        y_pred_unscaled_2d = scaler.inverse_transform(y_pred_2d)

        y_unscaled_mapped = y_unscaled_2d[:, node_map.cpu().numpy()]
        y_pred_unscaled_mapped = y_pred_unscaled_2d[:, node_map.cpu().numpy()]
        
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
