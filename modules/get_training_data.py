"""
Getting input data for model
"""
import os
from io import BytesIO
import pandas as pd
from zipfile import ZipFile
from urllib.request import urlopen
from config import training_data


def get_data():
    # Download Cisco Umbrella Popularity List (legit).
    if not os.path.isfile('input data/top-1m.csv'):
        resp = urlopen('http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip')
        zipfile = ZipFile(BytesIO(resp.read()))
        zipfile.extractall("input data")


def format_data():
    # Reduction to a unified form (extract second-level domain).
    for _ in training_data.items():
        domain_list = pd.read_csv('input data/top-1m.csv', names=['domain'])
        domain_list['domain'] = domain_list.applymap(lambda x: x.split('.')[0].strip().lower())
        domain_list['type'] = 'legit'
        training_data['legit'] = domain_list

        domain_list = pd.read_csv('input data/dga.csv', names=['domain'])
        domain_list['domain'] = domain_list.applymap(lambda x: x.split('.')[0].strip().lower())
        domain_list['type'] = 'dga'
        training_data['dga'] = domain_list


if __name__ == "__main__":
    get_data()
    format_data()
