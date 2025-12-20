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

        if json_file is None :
            raise ValueError("A json_file must be provided for the SupervisedAgent.")   
    
        if gym_type in [GYM_TYPE[ATTACKS_FROM_DATASET], GYM_TYPE[ATTACKS]]:
            X, y = self.init_attack_detection_env(json_file)
        elif gym_type in [GYM_TYPE[CLASSIFICATION_FROM_DATASET], GYM_TYPE[CLASSIFICATION]]:
            X, y = self.init_traffic_classification_env(json_file)
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
        df = pd.DataFrame(list(statuses))
        df.head()
        #df.info()

        del df['hostStatusesStructured']
        del df['status']
        del df['src_host']
        del df['dst_host']

        df = df.dropna()

        X = df.copy()
        y = X.pop('id')
        return X,y
        
    def init_attack_detection_env(self, json_file):
        statuses = read_data_file(json_file)
        df = pd.DataFrame(list(statuses))
        df.head()
        #df.info()
        df['is_attack'] = (df['id'] >= 2).astype(int)

        del df['hostStatusesStructured']
        del df['status']
        del df['id']

        df = df.dropna()

        X = df.copy()
        y = X.pop('is_attack')
        return X,y


    def predict(self, state):
        """
        receives normal state and predicts action
        """   
        row_data = pd.DataFrame({
            'packets': [state[0]],
            'bytes': [state[2]],
            'packetsPercentageChange': [state[1]],
            'bytesPercentageChange': [state[3]]
        })
        row_data_df = pd.DataFrame(row_data, index=[0])

        return self.clf.predict(row_data_df)
    
    def predict_attack(self, state):
        """
        receives normal state and predicts action
        """   
        #TODO adapt to global_state
        row_data = pd.DataFrame({
            'packets': [state[0]],
            'bytes': [state[2]],
            'packetsPercentageChange': [state[1]],
            'bytesPercentageChange': [state[3]]
        })
        row_data_df = pd.DataFrame(row_data, index=[0])
        
        # # Expected feature order for the classifier
        # try:
        #     row_data_df = row_data_df[self.expected_feature_order]
        # except KeyError as e:
        #     print(f"Errore: Una o più feature attese non sono presenti nel DataFrame di input: {e}")
        #     print(f"Feature attese: {self.expected_feature_order}")
        #     print(f"Feature nel DataFrame di input: {list(row_data_df.columns)}")
        #     raise # KeyError("Le feature del DataFrame di input non corrispondono a quelle attese dal classificatore.")

        return self.clf.predict(row_data_df)


