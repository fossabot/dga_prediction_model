"""
Starting capture network traffic
"""
import pickle
import tldextract
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from scapy.all import *
from scapy.layers.dns import DNS
from scapy.layers.inet import IP


# import logging
# logging.basicConfig(filename='example.log',level=logging.DEBUG)
# logging.getLogger("scapy").setLevel(1)

# ADD FUNCTION - DONT REPEAT THE SAME DOMAIN + logging to file

def packet_callback(packet):
    if IP in packet:
        ip_src = packet[IP].src
        ip_dst = packet[IP].dst
        if packet.haslayer(DNS) and packet.getlayer(DNS).qr == 0:
            qname = packet.getlayer(DNS).qd.qname.decode("utf-8")
            ext_qname = tldextract.extract(qname)
            if ext_qname.suffix != '' and ext_qname.suffix != 'localdomain' and len(ext_qname.domain) > 6:
                match = ngram_counts * vectorizer.transform([ext_qname.domain]).transpose()
                X_pred = [len(ext_qname.domain), match]
                if clf.predict([X_pred]) == 'dga':
                    print(str(ip_src.encode("utf-8")) + ' --> ' + str(ip_dst.encode("utf-8")) + ' : ' + '('
                          + qname + ')')


def capture():

    global ngram_counts
    global vectorizer
    global clf

    print("[*] Loading training dataset from disk...")
    with open('input data/training_data.pkl', 'rb') as f:
        training_data = pickle.load(f)

    print("[*] Loading model from disk...")
    clf = joblib.load('input data/model.pkl')

    print("[*] Loading and preparing necessary data for model...")
    vectorizer = CountVectorizer(ngram_range=(3, 5), analyzer='char', max_df=1.0, min_df=0.0001)
    ngram_matrix = vectorizer.fit_transform(training_data['legit']['domain'])
    ngram_counts = ngram_matrix.sum(axis=0).getA1()

    print("List system interfaces: ", os.listdir('/sys/class/net/'))
    interface = input("Enter desired interface: ")

    print("[*] Scanning...")

    start_time = datetime.now()

    sniff(iface=interface, filter="port 53", store=0, prn=packet_callback)

    stop_time = datetime.now()
    total_time = stop_time - start_time
    print("[*] Scan stopped")
    print("Scan duration: %s" % (total_time))


if __name__ == "__main__":
    capture()
