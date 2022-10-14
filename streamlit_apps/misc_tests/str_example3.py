import streamlit as st

import numpy as np 
import umap
import xgboost as xgb

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score



st.write("""
# Explore different classifier and datasets
Which one is the best?
""")

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest', 'XGBOOST')
)
def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == 'XGBOOST':
        clf = xgb.XGBClassifier(objective="binary:logistic", 
            n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
        
    return clf

clf = get_classifier(classifier_name, params)
#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

#### PLOT DATASET ####

# get the components in pca & t-sne
# Project the data onto the 2 primary principal components

scaler = StandardScaler()
# X = scaler.fit_transform(X) 

pca = PCA(2)
pca_projected = pca.fit_transform(X)

pca1 = pca_projected[:, 0]
pca2 = pca_projected[:, 1]

# Use a t-SNE plot now....
tsne = TSNE(n_components=2, verbose=0, random_state=123)
tsne_projected = tsne.fit_transform(X) 

ts1 = tsne_projected[:, 0]
ts2 = tsne_projected[:, 1]

um_reducer = umap.UMAP()
um_embedding = um_reducer.fit_transform(X)

um1 = um_embedding[:, 0]
um2 = um_embedding[:, 1]

plt.rcParams.update({'font.size': 7})
fig = plt.figure()
fig, ax = plt.subplots(3,1)

pca_points = ax[0].scatter(pca1, pca2,
        c=y, alpha=0.8,
        cmap='viridis')

ax[0].set_xlabel('Principal Component 1')
ax[0].set_ylabel('Principal Component 2')
fig.colorbar(pca_points, ax=ax[0])


ts_points = ax[1].scatter(ts1, ts2,
        c=y, alpha=0.8,
        cmap='viridis')

ax[1].set_xlabel('t-sne Component 1')
ax[1].set_ylabel('t-sne Component 2')
fig.colorbar(ts_points, ax=ax[1])

um_points = ax[2].scatter(um1, um2,
        c=y, alpha=0.8,
        cmap='viridis')

ax[2].set_xlabel('umap Component 1')
ax[2].set_ylabel('umap Component 2')
fig.colorbar(ts_points, ax=ax[2])


fig.tight_layout()
st.pyplot(fig)
