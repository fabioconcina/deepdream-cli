import argparse
import json
import logging
import os
from typing import Dict, List

from deep_dream import DeepDream

logger = logging.getLogger("deep_dream")


parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Deep Dreams with Keras. Multiple experiments.")
parser.add_argument("base_image_path", metavar="base_image_path", type=str, help="Path to the image to transform.")

args: argparse.Namespace = parser.parse_args()
base_image_path: str = args.base_image_path

with open("experiment.json") as json_file:
    config: Dict = json.load(json_file)
    experiment_list: List = config["experiment"]
    result_folder: str = config["result_folder"]

if not os.path.exists(result_folder):
    os.mkdir(result_folder)

for experiment in experiment_list:
    for experiment_name, experiment_dict in experiment.items():

        logger.info(f"*** Initiating experiment {experiment_name} ***")

        experiment_result_path: str = os.path.join(result_folder, experiment_name)
        experiment_dict["base_image_path"] = base_image_path
        experiment_dict["result_prefix"] = experiment_result_path
        with open(f"{experiment_result_path}.json", "w") as json_file:
            json.dump(experiment_dict, json_file)

        dream: DeepDream = DeepDream.from_dict(experiment_dict)
        dream.do_dream()
