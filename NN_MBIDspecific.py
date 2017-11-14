import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize


def classify(x,y):
    """
    This function consists of feeding data to the Logistic Regression Model,
    alongwith that it consists of calculating metrics that help's in analyzing
    the model calculates
    :param x: All the attributes barring labeled attribute
    :param y: The labeled attribute
    :return: None
    """
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=1)
    NN = MLPClassifier( hidden_layer_sizes = 10,learning_rate_init = 0.1)
    NN.fit(X_train, y_train)
    y_predict = NN.predict(X_test)
    acc = accuracy_score(y_test, y_predict) * 100
    print("The accuracy on running Neural Network achieved is", acc)
    print("The confusion Matrix is")
    print(confusion_matrix(y_test, y_predict))

    print("The precision,recall, f-score obtained for this model are")
    print(classification_report(y_test, y_predict))

    status = NN.predict_proba(X_test)
    y_test = label_binarize(y_test, classes=['normal.', 'smurf.','other'])

    samples = 3
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(samples):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], status[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[1], tpr[1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    roc_auc[1] = auc(fpr[1], tpr[1])
    print('The roc rate for 1st attribute is ', roc_auc[1])
    plt.show()


def main():
    """
    Main Function reads the data and removes useless attributes
    :return: None
    """

    data = pd.read_csv('misuseIDS_i_specific_final.csv')
    data.drop(['su_attempted','num_access_files','num_outbound_cmds','is_host_login'], axis = 1, inplace = True)
    x = data.drop('Result.',axis = 1)
    y = data['Result.']
    classify(x,y)

if __name__ == "__main__":
    main()