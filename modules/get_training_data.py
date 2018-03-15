"""
Downloads Cisco Umbrella Popularity List (legit), DGA-domains (results popular dga)
and extract second-level domain.
"""
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import os
import tldextract


def get_legit():
    if not os.path.isfile('../input data/top-1m.csv'):
        resp = urlopen('http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip')
        zipfile = ZipFile(BytesIO(resp.read()))
        zipfile.extractall("../input data")
    else:
        raw = open('../input data/top-1m.csv')
        for line in raw:
            return tldextract.extract(line.split(',')[1]).domain


def get_dga():
    if not os.path.isfile('../input data/dga.csv'):
        print("No dga training data.")
    else:
        raw = open('../input data/dga.csv')
        for line in raw:
            return tldextract.extract(line).domain
