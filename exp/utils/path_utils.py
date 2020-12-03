from datetime import datetime
import os
from typing import List, Dict, Any
import json


def create_time_folder(root_path: str, create: bool = True) -> str:
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = os.path.join(root_path, "rl", f"run_{now}")

    if create:
        create_if_not_exist(path)

    return path


def create_run_folder(
    root_path: str, algorthm_key: str, iteration: int, create: bool = True
):

    path = os.path.join(root_path, algorthm_key, str(iteration))

    if create:
        create_if_not_exist(path)

    return path


def create_if_not_exist(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_json_to_file(
    report_save_path: str, file_name: str, **kwargs,
):
    stringified = {k: str(v) for k, v in kwargs.items()}

    file_path = os.path.join(report_save_path, file_name)
    with open(file_path, "w") as file:
        json.dump(stringified, file)