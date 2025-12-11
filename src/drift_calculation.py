"""
Entrypoint for calculating drift monitoring metrics for Prometheus.
"""

import json
import math
import logging
import typing

import pandas as pd
from prometheus_client import Gauge, CollectorRegistry, push_to_gateway

from configs.configs import PathsConfig, TelemetryConfig, run_config

logger = logging.getLogger(__name__)

def get_psi(training_percentages: typing.Tuple[float], latest_percentages: typing.Tuple[float]) -> float:
    psi = 0
    for training, latest in zip(training_percentages, latest_percentages):
        psi += (latest - training) * math.log(latest / training, math.e)
    return psi

# TODO make monitoring metrics


def main():  # pragma: no cover
    with open(PathsConfig.telemetry_training_data_path, "r") as file:
        training_telemetry_data = json.load(file)
    with open(PathsConfig.telemetry_live_data_path, "r") as file:
        live_telemetry_data = json.load(file)
        live_telemetry_data = pd.DataFrame.from_records(live_telemetry_data)

    live_telemetry_data.sort_values("timestamp", inplace=True, ascending=False)
    latest_telemetry_data = live_telemetry_data.iloc[0 : TelemetryConfig.num_instances_for_live_dist]

    if telemetry_data_count := latest_telemetry_data.shape[0] < TelemetryConfig.num_instances_for_live_dist:
        logger.warning(
            f"Telemetry calculation has {telemetry_data_count} which is less than "
            f"required {TelemetryConfig.num_instances_for_live_dist}. This program will exit now."
        )
        exit(0)

    psi_s = {}
    for target in TelemetryConfig.targets:
        training_count = training_telemetry_data[target]["true"] + training_telemetry_data[target]["false"]
        # Adding Epsilon to avoid division by zero err for when there is no instance of a category.
        training_percentages = (
            (training_telemetry_data[target]["true"] / training_count) + TelemetryConfig.epsilon,
            (training_telemetry_data[target]["false"] / training_count) + TelemetryConfig.epsilon,
        )

        latest_count = latest_telemetry_data[target].shape[0]
        # Adding Epsilon to avoid division by zero err for when there is no instance of a category.
        latest_percentages = (
            (sum(latest_telemetry_data[target] == True) / latest_count) + TelemetryConfig.epsilon,
            (sum(latest_telemetry_data[target] == False) / latest_count) + TelemetryConfig.epsilon,
        )

        psi_s[target] = get_psi(training_percentages, latest_percentages)

    registry = CollectorRegistry()
    psi_gauge = Gauge(
        f"{run_config.app_name}_psi_s",
        "PSI calculations for feature and output data",
        labelnames=["target"],
        registry=registry,
    )
    for target in TelemetryConfig.targets:
        psi_gauge.labels(target=target).set(psi_s[target])

    push_to_gateway(TelemetryConfig.push_gateway_uri, job="telemetryBatch", registry=registry)


if __name__ == "__main__":
    main()