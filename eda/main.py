import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn.model_selection import cross_val_score

from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def main():
    st.title("An ML Look At Momentum/Reversal Trading Strategy Models")
    dataset_name = st.sidebar.selectbox("Select Dataset", ("j= 10 k= 10", "j= 10 k= 15", "j= 10 k= 20", "j= 15 k= 10", "j= 15 k= 15", "j= 15 k= 20", "j= 20 k= 20", 
    "j= 20 k= 40", "j= 30, k= 60"))
    st.write(dataset_name)

    def return_data(dataset):
        if dataset == "j= 10 k= 10":
            features = pd.read_csv('daily_j10_k10_features.csv')
            pct = pd.read_csv('daily_j10_k10_pct.csv')
        elif dataset == "j= 10 k= 15":
            features = pd.read_csv('daily_j10_k15_features.csv')
            pct = pd.read_csv('daily_j10_k15_pct.csv')
        elif dataset == "j= 10 k= 20":
            features = pd.read_csv('daily_j10_k20_features.csv')
            pct = pd.read_csv('daily_j10_k20_pct.csv')
        elif dataset == "j= 15 k= 10":
            features = pd.read_csv('daily_j15_k10_features.csv')
            pct = pd.read_csv('daily_j15_k10_pct.csv')
        elif dataset == "j= 15 k= 15":
            features = pd.read_csv('daily_j15_k15_features.csv')
            pct = pd.read_csv('daily_j15_k15_pct.csv')
        elif dataset == "j= 15 k= 20":
            features = pd.read_csv('daily_j15_k20_features.csv')
            pct = pd.read_csv('daily_j15_k20_pct.csv')
        elif dataset == "j= 20 k= 20":
            features = pd.read_csv('daily_j20_k20_features.csv')
            pct = pd.read_csv('daily_210_k20_pct.csv')
        elif dataset == "j= 20 k= 40":
            features = pd.read_csv('daily_j20_k40_features.csv')
            pct = pd.read_csv('daily_j20_k40_pct.csv')
        else:
            features = pd.read_csv('daily_j30_k60_features.csv')
            pct = pd.read_csv('daily_30_k60_pct.csv')

        num_colums = ['uint8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numerical_columns = list(features.select_dtypes(include=num_colums).columns)
        df = features[numerical_columns]
        return df


    def plot_metrics(metrics_list):

        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            predictions = clf.predict(X_test)
            matrix = confusion_matrix(y_test, predictions)
            st.write("Confusion Matrix ", matrix)

        if 'Classification_Report' in metrics_list:
            st.subheader('Classification_Report')
            predictions = clf.predict(X_test)
            report = classification_report(y_test, predictions)
            st.write("Classification_Report ", report)

        if 'Accuracy_Score' in metrics_list:
            st.subheader('Accuracy_Score')
            predictions = clf.predict(X_test)
            score = accuracy_score(y_test, predictions)
            st.write("Accuracy_Score: ", score.round(2))

    def split(df): 

        corr = df.corr() 
        correlated_features = set()
        for i in range(len(corr.columns)):
            for a in range(i):
                if abs(corr.iloc[i, a]) > 0.90:
                    colname = corr.columns[i]
                    correlated_features.add(colname)
        X_train, X_test, y_train, y_test = train_test_split(df.drop('label',axis=1), df['label'], train_size=0.33)
        X_before = X_test

        X_train.drop(columns=correlated_features, axis=1, inplace=True)
        X_test.drop(columns=correlated_features, axis=1, inplace=True)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test
    
    df = return_data(dataset_name)
    X_train, X_test, y_train, y_test = split(df)

    if st.sidebar.checkbox("show raw data", False):
        st.subheader("Data Frame from Observational Period J,K")
        st.write(df)
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "Classifier", ("Logistic Regression", "Decision Tree", "LSTM"))
    
    #Models 
    if classifier == "Logistic Regression":
        metrics = st.sidebar.multiselect("what metrics to plot?", ("Confusion Matrix", "Classification_Report", "Accuracy_Score"))
        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Logistic Regressor Classifier")
            param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
            clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
            GridSearchCV(cv=None,
                        estimator=LogisticRegression(C=1.0,intercept_scaling=1,   
                        dual=True, fit_intercept=True, penalty='l2'),
                        param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
            clf.fit(X_train,y_train)
            predictions = clf.predict(X_test)
            matrix = confusion_matrix(y_test, predictions)
            score = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)
            #st.write("Accuracy_Score: ", score.round(2))
            #st.write("Classification_Report ", report)
            #st.write("Confusion Matrix ", matrix)
            plot_metrics(metrics)

    if classifier == "Decision Tree":
        metrics = st.sidebar.multiselect("what metrics to plot?", ("Confusion Matrix", "Classification_Report", "Accuracy_Score"))
        if st.sidebar.button("Classify", key="classify"):
            scaler = StandardScaler()
            pca = decomposition.PCA()
            st.subheader("Decision Tree Classifier")
            dec_tree = tree.DecisionTreeClassifier()
            pipe = Pipeline(steps=[('std_slc', scaler),('pca', pca),('dec_tree', dec_tree)])
            n_components = list(range(1,X_train.shape[1]+1,1))
            criterion = ['gini', 'entropy']
            max_depth = [2,4,6,8,10,12]
            parameters = dict(pca__n_components=n_components, dec_tree__criterion=criterion, dec_tree__max_depth=max_depth)
            clf = GridSearchCV(pipe, parameters,n_jobs=-1,verbose=True)
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            matrix = confusion_matrix(y_test, predictions)
            score = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)
            #st.write("Accuracy_Score: ", score.round(2))
            #st.write("Classification_Report ", report)
            #st.write("Confusion Matrix ", matrix)
            plot_metrics(metrics)

    if classifier == "LSTM":
        metrics = st.sidebar.multiselect("what metrics to plot?", ("Confusion Matrix", "Classification_Report", "Accuracy_Score"))
        if st.sidebar.button("Classify", key="classify"):
            scaler = StandardScaler()
            pca = decomposition.PCA()
            st.subheader("Long-short term memory Model")
            dec_tree = tree.DecisionTreeClassifier()
            pipe = Pipeline(steps=[('std_slc', scaler),('pca', pca),('dec_tree', dec_tree)])
            n_components = list(range(1,X_train.shape[1]+1,1))
            criterion = ['gini', 'entropy']
            max_depth = [2,4,6,8,10,12]
            parameters = dict(pca__n_components=n_components, dec_tree__criterion=criterion, dec_tree__max_depth=max_depth)
            clf = GridSearchCV(pipe, parameters,n_jobs=-1,verbose=True)
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            matrix = confusion_matrix(y_test, predictions)
            score = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)
            #st.write("Accuracy_Score: ", score.round(2))
            #st.write("Classification_Report ", report)
            #st.write("Confusion Matrix ", matrix)
            plot_metrics(metrics)
    
if __name__ == '__main__':
    main()






















