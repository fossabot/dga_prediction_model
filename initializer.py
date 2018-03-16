from modules.get_training_data import *

"""Get training data"""
get_data()
training_data = format_data()

""" Overall data """
all_data_dict = pd.concat([training_data['legit'], training_data['dga']], ignore_index=True)
print(all_data_dict)