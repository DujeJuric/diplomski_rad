import torch
from flwr.client import ClientApp, NumPyClient

from task import prepare_model, load_flower_data, train_online, test

class STGCNClient(NumPyClient):
    def __init__(self, partition_id, num_partitions, run_config):
        self.partition_id = partition_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = run_config["dataset"]
        self.batch_size = run_config["batch-size"]
        self.online_steps = run_config["online-steps"]
        self.locations_json_path = run_config["locations-json-path"]
        self.cloudlet_experiment = run_config["cloudlet-experiment"]
        self.dataset_path = run_config["dataset-path"]
        
        self.model = prepare_model(self.dataset, self.dataset_path, self.device)
        self.x_train, self.y_train, self.end_of_initial_data_index, self.data_per_step, self.val_iter, self.node_map = load_flower_data(
            self.dataset,
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
        
        epochs = config["local-epochs"]
        lr = config["learning-rate"]
        
        train_loss = train_online(self.model, self.x_train, self.y_train, self.end_of_initial_data_index, self.data_per_step, self.node_map, epochs, lr, self.batch_size, self.online_steps, self.device, self.partition_id)
        
        return self.get_parameters(config={}), self.x_train.shape[0], {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        loss = test(self.model, self.val_iter, self.node_map)
        
        return float(loss), len(self.val_iter.dataset), {"eval_loss": float(loss)}

def client_fn(context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.run_config["num-partitions"]
    run_config = context.run_config
    return STGCNClient(partition_id, num_partitions, run_config).to_client()

app = ClientApp(client_fn=client_fn)
