from numpy import ravel_multi_index
from sklearn import datasets
from util import knn
from sklearn.model_selection import train_test_split
from util import accuracy_score as accuracy

iris=datasets.load_iris()

x, y = iris.data, iris.target

X_train, X_test, Y_train, Y_test=train_test_split(x, y, test_size=0.2, random_state=1)

k=int(input("enter the k value\n"))
clf=knn(k=k)
clf.fit(X_train, Y_train)
predict=clf.predict(X_test)
print("Knn classification accuracy is", accuracy(Y_test, predict))