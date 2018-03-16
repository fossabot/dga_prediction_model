#!/usr/bin/env bash
apt update -y && apt upgrade -y
apt install python3, python3-pip -y
pip3 install -r requirements.txt