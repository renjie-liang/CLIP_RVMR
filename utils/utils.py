import os
import json
import zipfile
import numpy as np
import pickle
import yaml



def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))
