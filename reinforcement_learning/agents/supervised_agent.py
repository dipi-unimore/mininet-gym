"""
Classification 
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utility.constants import ATTACKS, ATTACKS_FROM_DATASET, CLASSIFICATION_FROM_DATASET, CLASSIFICATION, GYM_TYPE, MARL_ATTACKS, MARL_ATTACKS_FROM_DATASET
from utility.my_files import read_data_file

class SupervisedAgent:
    def __init__(self, gym_type, json_file = None):

        self.is_classification_gym_type = False

        if json_file is None :
            raise ValueError("A json_file must be provided for the SupervisedAgent.")   
    
        if gym_type in [GYM_TYPE[ATTACKS_FROM_DATASET], GYM_TYPE[ATTACKS]]:
            X, y = self.init_attack_detection_env(json_file)
        elif gym_type in [GYM_TYPE[CLASSIFICATION_FROM_DATASET], GYM_TYPE[CLASSIFICATION]]:
            X, y = self.init_traffic_classification_env(json_file)
            self.is_classification_gym_type = True
        elif gym_type in [GYM_TYPE[MARL_ATTACKS_FROM_DATASET], GYM_TYPE[MARL_ATTACKS]]:
            X, y = self.init_marl_attack_env(json_file)
        else:
            raise ValueError(f"Unsupported gym_type for SupervisedAgent: {gym_type}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        self.clf = DecisionTreeClassifier(max_depth=4)
        self.clf.fit(X_train, y_train)
        
        y_pred = self.clf.predict(X_test)
        
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision, self.recall, self.fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

   
    def init_traffic_classification_env(self, json_file):
        statuses = read_data_file(json_file)
        new_statuses = []
        for status in statuses:
            new_status = {}
            new_status['id'] = status['id']
            new_status['received_packets'] = sum(host['receivedPackets'] for host in status['hostStatusesStructured'].values())
            new_status['received_bytes'] = sum(host['receivedBytes'] for host in status['hostStatusesStructured'].values())
            new_status['transmitted_packets'] = sum(host['transmittedPackets'] for host in status['hostStatusesStructured'].values())   
            new_status['transmitted_bytes'] = sum(host['transmittedBytes'] for host in status['hostStatusesStructured'].values()) 
            new_statuses.append(new_status)
        
        df = pd.DataFrame(list(new_statuses))
        df.head()

        df = df.dropna()

        X = df.copy()
        y = X.pop('id')
        return X,y
        
    def init_attack_detection_env(self, json_file):
        statuses = read_data_file(json_file)
        new_statuses = []
        for status in statuses:
            new_status = {}
            new_status['is_attack'] = int(status['id'] >= 2)
            new_status['packets'] = status['packets']
            new_status['packets_percentage_change'] = status['packetsPercentageChange'] 
            new_status['bytes'] = status['bytes']
            new_status['bytes_percentage_change'] = status['bytesPercentageChange']  
            new_statuses.append(new_status)
        
        df = pd.DataFrame(list(new_statuses))
        df.head()
        #df.info()

        df = df.dropna()

        X = df.copy()
        y = X.pop('is_attack')
        return X,y

    def train(self, statuses):
        """
        trains the model with new data
        """  
        # Convert list of statuses to DataFrame
        new_statuses = []
        for status in statuses:
            new_status = {}
            new_status['is_attack'] = int(status['id'] >= 2)
            new_status['packets'] = status['packets']
            new_status['packets_percentage_change'] = status['packetsPercentageChange'] 
            new_status['bytes'] = status['bytes']
            new_status['bytes_percentage_change'] = status['bytesPercentageChange']  
            new_statuses.append(new_status)
        df = pd.DataFrame(new_statuses)

        X = df[['packets', 'packets_percentage_change', 'bytes', 'bytes_percentage_change']]
        y = df['is_attack']

        # Fit the model on the new data
        self.clf.fit(X, y)

    def predict(self, state):
        """
        receives normal state and predicts action
        """  
        if not self.is_classification_gym_type: #attacks, marl attacks
            row_data = pd.DataFrame({
                'packets': [state[0]],
                'packets_percentage_change': [state[1]],
                'bytes': [state[2]],
                'bytes_percentage_change': [state[3]]
            })
        else:
            row_data = pd.DataFrame({
                'received_packets': [state[0]],
                'received_bytes': [state[2]],
                'transmitted_packets': [state[1]],
                'transmitted_bytes': [state[3]]
            })
        row_data_df = pd.DataFrame(row_data, index=[0])

        return self.clf.predict(row_data_df)
    




