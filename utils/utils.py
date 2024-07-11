import os
import json
import zipfile
import numpy as np
import pickle
import yaml
import time
import logging
import os


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





def get_logger(dir, tile):
    os.makedirs(dir, exist_ok=True)
    log_file = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(dir, "{}_{}.log".format(log_file, tile))

    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(levelname)s:%(message)s"
    # DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)

    fhlr = logging.FileHandler(log_file) 
    fhlr.setFormatter(formatter)
    fhlr.setLevel('INFO') 

    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger



class LossTracker:
    def __init__(self):
        self.total_loss = 0.0
        self.num_steps = 0

    def update(self, loss):
        self.total_loss += loss
        self.num_steps += 1

    def average_loss(self):
        if self.num_steps == 0:
            return 0
        return self.total_loss / self.num_steps

    def reset(self):
        self.total_loss = 0.0
        self.num_steps = 0
        
        
class TimeTracker:
    def __init__(self):
        self.times = {}
        self.start_times = {}

    def start(self, name):
        self.start_times[name] = time.time()

    def stop(self, name):
        if name not in self.times:
            self.times[name] = 0
        if name in self.start_times:
            self.times[name] += time.time() - self.start_times[name]
            del self.start_times[name]

    def get_time(self, name):
        return self.times.get(name, 0)

    def reset(self, name):
        if name in self.times:
            self.times[name] = 0

    def reset_all(self):
        self.times = {}
        self.start_times = {}

    def report(self):
        report = "\n".join([f"{name}: {time:.4f} seconds" for name, time in self.times.items()])
        return report


