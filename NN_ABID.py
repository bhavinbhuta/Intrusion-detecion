"""
This file a specimen of implementing Neural Network on Anomaly Based Intrusion
Data, the data provided to learn the model is from Kdd dataset. Various metrics
are calculated to evaluate the model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def classify(x,y):
    """
    This function consists of feeding data to the Neural Network Model,
    alongwith that it consists of calculating metrics that help's in analyzing
    the model calculates
    :param x: All the attributes barring labeled attribute
    :param y: The labeled attribute
    :return: None
    """
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=2)
    NN = MLPClassifier(hidden_layer_sizes = 8000,learning_rate_init = 0.1)
    NN.fit(X_train, y_train)
    y_predict = NN.predict(X_test)
    print('Results obtained by running Neural Network on Anamoly Based IDS')

    acc = accuracy_score(y_test, y_predict) * 100
    print("The accuracy on running Neural Network achieved is", acc)

    print("The confusion Matrix is")
    print(confusion_matrix(y_test, y_predict))

    print("The precision,recall, f-score obtained for this model are")
    print(classification_report(y_test, y_predict))

    status = NN.predict_proba(X_test)

    fpr, tpr, _ = roc_curve(y_test, status[:, 1], pos_label='normal')
    roc_auc = auc(fpr, tpr)
    print('The roc rate is ', roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr, tpr, label='RF')
    plt.show()

def main():
    """
    Main Function reads the data and removes useless attributes
    :return: None
    """
    data = pd.read_csv('kdddataset_ikp.csv')
    data.drop(['land', 'num_outbound_cmds', 'is_host_login', 'urgent'], axis=1,
              inplace=True)
    x = data.drop('Result', axis=1)
    y = data['Result']
    classify(x, y)

if __name__ == "__main__":
    main()