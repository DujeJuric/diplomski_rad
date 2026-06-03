from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

def fit_metrics_aggregation_fn(all_client_metrics):
    total_examples = sum([num_examples for num_examples, _ in all_client_metrics])
    if total_examples == 0:
        return {}
    
    total_loss = sum(
        [num_examples * metrics.get("train_loss", 0.0) for num_examples, metrics in all_client_metrics]
    )
    return {"train_loss": total_loss / total_examples}

def evaluate_metrics_aggregation_fn(all_client_metrics):
    total_examples = sum([num_examples for num_examples, _ in all_client_metrics])
    if total_examples == 0:
        return {}
    
    total_loss = sum(
        [num_examples * metrics.get("eval_loss", 0.0) for num_examples, metrics in all_client_metrics]
    )
    return {"eval_loss": total_loss / total_examples}

def server_fn(context):
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    num_partitions = context.run_config["num-partitions"]
    
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    def fit_config(server_round: int):
        return {
            "local-epochs": local_epochs,
            "learning-rate": learning_rate,
        }

    strategy = FedAvg(
        fraction_fit=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=num_partitions,
        min_evaluate_clients=num_partitions,
        min_available_clients=num_partitions,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=num_rounds))

app = ServerApp(server_fn=server_fn)
