"""
Starting capture network traffic
"""
import sys
import os
from datetime import datetime

# from scapy.layers.dns import DNS
# from scapy.layers.inet import IP
from sklearn.externals import joblib
# import tldextract
# from scapy.all import *
# from scapy import packet
# from scapy.sendrecv import sniff


# def sniff_packets(packet):
#     if IP in packet:
#         pckt_src = packet[IP].src
#         pckt_dst = packet[IP].dst


def capture():

    clf = joblib.load('../input data/model.pkl')

    # print("List system interfaces: ", os.listdir('/sys/class/net/'))
    # interface = input("Enter desired interface: ")

    # start_time = datetime.now()

    print("[*] Scanning...")
    clf('')



    # sniff(iface=interface, filter="udp and port 53", store=0, prn=sniff_packets)

    # stop_time = datetime.now()
    # total_time = stop_time - start_time

    # print("[*] Scan stopped")
    # print("Scan duration: %s" %(total_time))


if __name__ == "__main__":
    capture()