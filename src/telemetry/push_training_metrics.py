import time
import numpy as np
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from configs.configs import TelemetryConfig, run_config

registry= CollectorRegistry()

def push_last_training_time():  # pragma: no cover


    last_training_gauge = Gauge( 
                'alcoholerometer_last_train_timestamp',
                'Timestamp of the last successful training',
                registry=registry
            )
    last_training_gauge.set(time.time())

    push_to_gateway(TelemetryConfig.push_gateway_uri, job=f"{run_config.app_name}_push_training_metrics", registry=registry)

def push_model_accuracy(accuracy: float):  # pragma: no cover

    mean_accuracy = np.mean(accuracy)

    model_accuracy_gauge = Gauge( 
                'alcoholerometer_model_accuracy',
                'Accuracy of the trained model',
                registry=registry
            )
    model_accuracy_gauge.set(mean_accuracy)

    push_to_gateway(TelemetryConfig.push_gateway_uri, job=f"{run_config.app_name}_push_training_metrics", registry=registry)