import torch
from flwr.client import ClientApp, NumPyClient

from task import prepare_model, load_flower_data, train_online, test

class STAEformerClient(NumPyClient):
    def __init__(self, partition_id, num_partitions, run_config):
        self.partition_id = partition_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = run_config.get("dataset", "data_PEMSD7")
        self.batch_size = run_config.get("batch-size", 16)
        self.online_steps = run_config["online-steps"]
        self.locations_json_path = run_config["locations-json-path"]
        self.cloudlet_experiment = run_config["cloudlet-experiment"]
        self.dataset_path = run_config["dataset-path"]
        
        self.model = prepare_model(self.dataset_path, self.device)
        self.x_train, self.y_train, self.end_of_initial_data_index, self.data_per_step, self.val_iter, self.node_map = load_flower_data(
            partition_id,
            num_partitions,
            self.batch_size,
            self.online_steps,
            self.locations_json_path,
            self.cloudlet_experiment,
            self.dataset_path,
            device=self.device
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        epochs = config.get("local-epochs", 1)
        lr = config.get("learning-rate", 0.001)
        
        train_loss = train_online(self.model, self.x_train, self.y_train, self.end_of_initial_data_index, self.data_per_step, self.node_map, epochs, lr, self.batch_size, self.online_steps, self.device, self.partition_id)
        
        return self.get_parameters(config={}), self.x_train.shape[0], {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        loss = test(self.model, self.val_iter, self.node_map)
        
        return float(loss), len(self.val_iter.dataset), {"eval_loss": float(loss)}

def client_fn(context):
    partition_id = context.node_config.get("partition-id", 0)
    num_partitions = context.run_config.get("num-partitions", 4)
    run_config = context.run_config
    return STAEformerClient(partition_id, num_partitions, run_config).to_client()

app = ClientApp(client_fn=client_fn)
