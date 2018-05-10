# DGA Prediction Model
#### Prediction Model based on frequency analysis of ngrams with machine learning (used Random Forest). The algorithm is able to detect dga-domains in the general active flow of dns-requests. Also, if necessary, can block suspicious hosts by filtering their packets.

Installation
-
##### Clone the DGA Prediction Model repository and enter the directory.
``` 
$ git clone https://github.com/exctzo/dga_prediction_model.git && cd dga_prediction_model
```
##### Execute the install.sh
```
$ chmod +x install.sh
$ ./install.sh
```


Usage
-
##### Launch the initialization.py script. Then follow the prompts in MENU to choose point to execute.
```
$ python3 initializer.py
```

#### In addition:
##### Blocking of possible dangerous hosts occurs by adding rules to iptables tables. The host can be blocked after the value of the dga-requests sent to server becomes more than 10 during scan traffic.
##### To view iptables rules:
```
$ iptables -L
```
##### The rules apply to the current server session, after the reboot the tables are cleared.
##### To immediately delete the rule, use the command:
```
$ iptables -D INPUT -s ip_host_address -j DROP
```


DGA-domains
-
##### Repo contain file with the results of popular DGA (original file from the [repo](https://github.com/andrewaeva/DGA)). Kept the same [GNU General Public license v2](http://opensource.org/licenses/gpl-2.0.php).