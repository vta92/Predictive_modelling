#python 3
import numpy as np
import sklearn.datasets
import sklearn.svm
import sklearn.model_selection

wine = sklearn.datasets.load_wine()

#non k-fold, 20:80 split
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        wine.data,wine.target,train_size = 0.8)
SVC = sklearn.svm.SVC(C=1.1,kernel='linear')
SVC.fit(x_train,y_train)
model = SVC.predict(x_test)
scores = SVC.score(x_test, y_test)
print(scores)
#92% accuracy


#k-fild, 5 buckets
#97.2% accuracy
cv = sklearn.model_selection.cross_val_score(SVC,wine.data, wine.target, cv=5)
print(cv)
print(cv.mean())