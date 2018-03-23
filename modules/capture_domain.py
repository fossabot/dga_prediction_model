"""
Starting capture network traffic
"""
# import sys
# import os
# from datetime import datetime

# from scapy.layers.dns import DNS
# from scapy.layers.inet import IP
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
# from config import training_data
import pickle
# import tldextract
# from scapy.all import *
# from scapy import packet
# from scapy.sendrecv import sniff


# def sniff_packets(packet):
#     if IP in packet:
#         pckt_src = packet[IP].src
#         pckt_dst = packet[IP].dst


def capture():

    print("[*] Loading training dataset from disk...")
    with open('input data/training_data.pkl', 'rb') as f:
        training_data = pickle.load(f)

    print("[*] Loading model from disk...")
    clf = joblib.load('input data/model.pkl')

    print("[*] Loading and preparing necessary data for model...")
    vectorizer = CountVectorizer(ngram_range=(3, 5), analyzer='char', max_df=1.0, min_df=0.0001)
    ngram_matrix = vectorizer.fit_transform(training_data['legit']['domain'])
    ngram_counts = ngram_matrix.sum(axis=0).getA1()

    domain = 'otqobfxqpyfxo'
    match = ngram_counts * vectorizer.transform([domain]).transpose()
    X_pred = [len(domain), match]
    print(domain, clf.predict([X_pred]))

    # print("List system interfaces: ", os.listdir('/sys/class/net/'))
    # interface = input("Enter desired interface: ")

    # start_time = datetime.now()

    # print("[*] Scanning...")

    # sniff(iface=interface, filter="udp and port 53", store=0, prn=sniff_packets)

    # stop_time = datetime.now()
    # total_time = stop_time - start_time

    # print("[*] Scan stopped")
    # print("Scan duration: %s" %(total_time))


if __name__ == "__main__":
    capture()
