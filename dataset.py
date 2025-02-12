import pandas as pd
import json

def get_jsonl_data(file_path):
    # Read the JSONL file into a Pandas DataFrame
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    # Load into Pandas DataFrame
    df = pd.DataFrame(data)

    return df

def construct_prompts(input_file, inst):
    with open(input_file, "r") as f:
        samples = f.readlines()
    samples = [json.loads(sample) for sample in samples]
    prompts = []
    for sample in samples:
        key = sample["project"] + "_" + sample["commit_id"]
        p = {"sample_key": key}
        p["func"] = sample["func"]
        p["target"] = sample["target"]
        p["prompt"] = inst.format(func=sample["func"])
        prompts.append(p)
    return prompts