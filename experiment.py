import argparse
import json
import os
from typing import Dict, List

from deep_dream import DeepDream


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
        print(f"\n*** Initiating experiment {experiment_name}")
        experiment_dict["base_image_path"] = base_image_path
        experiment_dict["result_prefix"] = os.path.join(result_folder, experiment_name)
        dream: DeepDream = DeepDream.from_dict(experiment_dict)
        dream.do_dream()
