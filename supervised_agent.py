"""
Classification 
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utility.my_files import read_data_file

class SupervisedAgent:
    def __init__(self, csv_file = 'traffic.csv'): 

        df = pd.read_csv(csv_file)
        df.head()
        #df.info()

        del df['i_src_host']
        del df['i_dst_host']
        df = df.dropna()

        X = df.copy()
        y = X.pop('traffic_type')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        self.clf = DecisionTreeClassifier(max_depth=4)
        self.clf.fit(X_train, y_train)
        self.row_to_predict = X_test.iloc[[110]]
        y_pred = self.clf.predict(X_test)
        
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision, self.recall, self.fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

    def __init__(self): 
        statuses = read_data_file('statuses')
        df = pd.DataFrame(list(statuses))
        df.head()
        #df.info()
        df['is_attack'] = (df['id'] >= 2).astype(int)

        del df['text']
        del df['id']

        df = df.dropna()

        X = df.copy()
        y = X.pop('is_attack')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        self.clf = DecisionTreeClassifier(max_depth=4)
        self.clf.fit(X_train, y_train)
        self.row_to_predict = X_test.iloc[[110]]
        y_pred = self.clf.predict(X_test)
        
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision, self.recall, self.fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')


    def predict(self, state):
        """
        receives normal state and predicts action
        """   
        row_data = pd.DataFrame({
            'packets_received': [state[0]],
            'bytes_received': [state[2]],
            'packets_transmitted': [state[1]],
            'bytes_transmitted': [state[3]]
        })
        row_data_df = pd.DataFrame(row_data, index=[0])

        return self.clf.predict(row_data_df)
    
    def predict_attack(self, state):
        """
        receives normal state and predicts action
        """   
        row_data = pd.DataFrame({
            'variation_packet': [state[1]],
            'variation_byte': [state[3]],
            'packets': [state[0]],
            'bytes': [state[2]]
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


