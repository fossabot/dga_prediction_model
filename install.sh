#!/usr/bin/env bash
sudo apt update -y && apt upgrade -y
sudo apt install python3 -y
sudo apt install python3-pip -y
sudo pip3 install -r requirements.txt