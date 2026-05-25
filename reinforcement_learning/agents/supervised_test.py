"""
Classification 
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

df = pd.read_csv('traffic.csv')
df.head()
df.info()

del df['i_src_host']
del df['i_dst_host']
df = df.dropna()
df.info()

import seaborn as sns
sns.boxplot(data=df.loc[:, ['packets_received', 'bytes_received', 'traffic_type']], x='packets_received', y='bytes_received', hue='traffic_type')

import numpy as np
# Calculate the Z-score for each value in the DataFrame
from scipy.stats import zscore

z_scores = np.abs(zscore(df))

# Set threshold for Z-score
threshold = 3

# Find the outliers
outliers = (z_scores > threshold)
print("Outliers detected:\n", df[outliers.any(axis=1)])

# Calculate Q1 and Q3
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Define outliers based on IQR
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))

# Print the outliers
print("Outliers detected:\n", df[outliers.any(axis=1)])

X = df.copy()
y = X.pop('traffic_type')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(y)

print(X_train.size, X_test.size)

print(y_test)

clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X_train, y_train)

from matplotlib import pyplot as plt
from sklearn.tree import plot_tree


# Set the figure size and DPI for higher resolution
plt.figure(figsize=(12, 8), dpi=300)  # You can adjust the width and height

# Plot the tree
plot_tree(clf, filled=True)

# Show the plot
plt.show()

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

precision_recall_fscore_support(y_test, y_pred, average='macro')

row_to_predict = X_test.iloc[[110]]
y_pred = clf.predict(row_to_predict)
print(X_test.iloc[[110]] )
print(y_test.iloc[[110]] )
print(y_pred)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

#regressione
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
dfr = pd.read_csv('traffic.csv')
del dfr['i_src_host']
del dfr['i_dst_host']
dfr = dfr.dropna()
X = df.iloc[:, :-1]
X.head()
y = df.iloc[:, -1]
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(X_train.size, X_test.size)
reg = LassoCV(cv=2, random_state=0).fit(X_train, y_train)  # addestriamo il modello
y_pred = reg.predict(X_test)  # facciamo la previsione
y__test_pred = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
sns.lineplot(data=y__test_pred)

row_to_predict = X_test.iloc[[90]]
y_pred = clf.predict(row_to_predict)
print(X_test.iloc[[90]] )
print(y_test.iloc[[90]] )
print(y_pred)

#clustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
dfc = pd.read_csv('traffic.csv', na_values='?')
del dfc['i_src_host']
del dfc['i_dst_host']
dfc = dfc.dropna()
pca = PCA(n_components=2)
df_reduced = pca.fit_transform(dfc)
df_reduced = df_reduced.transpose()
print(df_reduced)
df_reduced = {'col1': df_reduced[0], 'col2': df_reduced[1]}
df_reduced = pd.DataFrame(df_reduced)
sns.scatterplot(x='col1', y='col2', data=df_reduced)
clusters = KMeans(n_clusters=5).fit_predict(df_reduced)
df_reduced_clust = df_reduced.copy()
df_reduced_clust['cluster'] = clusters

