"""
Classification 
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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


