from numpy import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pickle

def loadData(filename):
    observations = pd.read_csv(filename)
    Y = observations.iloc[:, -1]
    X = observations.iloc[:, 0:19]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test


def classifier(clfname, clf, X_train, X_test, Y_train, Y_test, save_to=''):
    print(clfname)
    clf.fit(X_train, Y_train)
    if save_to != "":
        pickle.dump(clf, open(save_to, 'wb'))
    Predict = (clf.predict(X_test))
    target_names = ['class 0', 'class 1']
    confusion_matrix(Y_test, Predict)
    return(classification_report(Y_test, Predict, target_names = target_names))

"""

test_definition = {
    'nearest_neighbor': {
        'process': True,
        'model': KNeighborsClassifier(n_neighbors=5, weights='distance')
    },
    'decision_tree': {
        'process': True,
        'model': tree.DecisionTreeClassifier(criterion='log_loss', max_features='sqrt')
    },
    'logistic_regression': {
        'process': True,
        'model': LogisticRegression(max_iter=35000, C=5, solver='liblinear')
    },
    'support_verctor': {
        'process': True,
        'model': svm.SVC(C=10, kernel='linear')
    },
    'naive_bayes': {
        'process': True,
        'model': GaussianNB()
    }
}

test_data = {
    'Balanced': "combined_joro_v2.csv",
    'Imbalanced': "combined_joro_imbalanced.csv"
}

important_features = ['MeanDiurnalRange', 'Isothermality', 'TempAnnRange','MeanTempWetQtr', 'AnnPercip', 'PrecipWetQtr', 'PrecipWrmQtr']

for obs_name, file_name in test_data.items():
    print(obs_name, file_name)
    X_train, X_test, Y_train, Y_test = loadData(file_name)
    for model_name, mp in test_definition.items():
        if mp['process'] == False:
            continue
        print(classifier(model_name, mp['model'], X_train, X_test, Y_train, Y_test))

        X_train = X_train[['MeanDiurnalRange', 'Isothermality', 'TempAnnRange','MeanTempWetQtr', 'AnnPercip', 'PrecipWetQtr', 'PrecipWrmQtr']]
        X_test = X_test[['MeanDiurnalRange', 'Isothermality', 'TempAnnRange','MeanTempWetQtr', 'AnnPercip', 'PrecipWetQtr', 'PrecipWrmQtr']]
        print(classifier(f"{model_name} w/importance", mp['model'], X_train, X_test, Y_train, Y_test))

"""

## Taken from: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
# Save the model
X_train, X_test, Y_train, Y_test = loadData('combined_joro_v2.csv')
print(classifier('decision tree', tree.DecisionTreeClassifier(), X_train, X_test, Y_train, Y_test, save_to="decision_tree_model.sav"))
# Load and test the model
loaded_model = pickle.load(open("decision_tree_model.sav", 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)


"""

X_train, X_test, Y_train, Y_test = loadData('combined_joro_v2.csv')
print(classifier('kNN', KNeighborsClassifier(n_neighbors=5), X_train, X_test, Y_train, Y_test))
print(classifier('decision tree', tree.DecisionTreeClassifier(), X_train, X_test, Y_train, Y_test))
print(classifier('Logistic Regression', LogisticRegression(random_state=0, max_iter=35000), X_train, X_test, Y_train, Y_test))
print(classifier('SVC', SVC(), X_train, X_test, Y_train, Y_test))
print(classifier('Naive Bayes', GaussianNB(), X_train, X_test, Y_train, Y_test))

X_train, X_test, Y_train, Y_test = loadData('combined_joro_unbalanced.csv')
print(classifier('kNN', KNeighborsClassifier(n_neighbors=5), X_train, X_test, Y_train, Y_test))
print(classifier('decision tree', tree.DecisionTreeClassifier(), X_train, X_test, Y_train, Y_test))
print(classifier('Logistic Regression', LogisticRegression(random_state=0, max_iter=35000), X_train, X_test, Y_train, Y_test))
print(classifier('SVC', SVC(), X_train, X_test, Y_train, Y_test))
print(classifier('Naive Bayes', GaussianNB(), X_train, X_test, Y_train, Y_test))

print('Permutation Importance - Random Forest using permutation, values above 0.2')
X_train, X_test, Y_train, Y_test = loadData('combined_joro_v2.csv')
#X_train = X_train[['Annual_mean_temp', 'PrecipDryQtr', 'PercipSeasonality', 'MeanTempWetQtr', 'PercipDryMo']]
#X_test = X_test[['Annual_mean_temp', 'PrecipDryQtr', 'PercipSeasonality', 'MeanTempWetQtr', 'PercipDryMo']]
X_train = X_train[['MeanDiurnalRange', 'Isothermality', 'TempAnnRange','MeanTempWetQtr', 'AnnPercip', 'PrecipWetQtr', 'PrecipWrmQtr']]
X_test = X_test[['MeanDiurnalRange', 'Isothermality', 'TempAnnRange','MeanTempWetQtr', 'AnnPercip', 'PrecipWetQtr', 'PrecipWrmQtr']]

print(classifier('kNN', KNeighborsClassifier(n_neighbors=5), X_train, X_test, Y_train, Y_test))
print(classifier('decision tree', tree.DecisionTreeClassifier(), X_train, X_test, Y_train, Y_test))
print(classifier('Logistic Regression', LogisticRegression(random_state=0, max_iter=35000), X_train, X_test, Y_train, Y_test))
print(classifier('SVC', SVC(), X_train, X_test, Y_train, Y_test))
print(classifier('Naive Bayes', GaussianNB(), X_train, X_test, Y_train, Y_test))

print('Feature Importance - Chi-Squared')
X_train, X_test, Y_train, Y_test = loadData('combined_joro_v2.csv')
X_train = X_train[['Annual_mean_temp', 'PrecipDryQtr', 'PercipSeasonality', 'MeanTempWetQtr', 'PercipDryMo']]
X_test = X_test[['Annual_mean_temp', 'PrecipDryQtr', 'PercipSeasonality', 'MeanTempWetQtr', 'PercipDryMo']]


print(classifier('kNN', KNeighborsClassifier(n_neighbors=5), X_train, X_test, Y_train, Y_test))
print(classifier('decision tree', tree.DecisionTreeClassifier(), X_train, X_test, Y_train, Y_test))
print(classifier('Logistic Regression', LogisticRegression(random_state=0, max_iter=35000), X_train, X_test, Y_train, Y_test))
print(classifier('SVC', SVC(), X_train, X_test, Y_train, Y_test))
print(classifier('Naive Bayes', GaussianNB(), X_train, X_test, Y_train, Y_test))

"""