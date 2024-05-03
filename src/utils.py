import os
import joblib
import yaml


def config():
    with open("./config.yml", "r") as file:
        config = yaml.safe_load(file)

    return config


def dump(value=None, filename=None):
    if (value is not None) and (filename is not None):
        joblib.dump(value=value, filename=filename)

    else:
        raise ValueError("Value or filename is not found".capitalize())


def load(filename=None):
    if filename is not None:
        return joblib.load(filename=filename)

    else:
        raise ValueError("Filename is not found".capitalize())
