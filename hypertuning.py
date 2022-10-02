from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
import pandas as pd

iris = datasets.load_iris()

# How to select values https://www.youtube.com/watch?v=jUxhUgkKAjE
clf = GridSearchCV(svm.SVC(gamma='auto'), {
    'C': [1,10,20],
    'kernel': ['rbf', 'linear']
}, return_train_score=False)

clf.fit(iris.data, iris.target)

df = pd.DataFrame(clf.cv_results_)

print(df)




