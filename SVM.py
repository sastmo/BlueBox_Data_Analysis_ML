# https://towardsdatascience.com/support-vector-machines-svm-c9ef22815589
# https://chat.openai.com/share/b884a205-e245-4511-8742-be7784ab54d5

# Linear SVC *****************************************************************************************

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

x = np.array([[1, 3], [1, 2], [1, 1.5], [1.5, 2], [2, 3], [2.5, 1.5],
              [2, 1], [3, 1], [3, 2], [3.5, 1], [3.5, 3]])

y = [0] * 6 + [1] * 5

# Original Data Scatter Plot
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, alpha=0.9)
plt.show()

# Linear SVC with Default C
svc = svm.SVC(kernel='linear').fit(x, y)
X, Y = np.mgrid[0:4:200j, 0:4:200j]
Z = svc.decision_function(np.c_[X.ravel(), Y.ravel()])
Z = Z.reshape(X.shape)

# Plotting the decision boundary
plt.contourf(X, Y, Z > 0, alpha=0.4)
plt.contour(X, Y, Z, colors=['k'], linestyles=['-'], levels=[0])
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, alpha=0.9)
plt.show()

# Linear SVC with C=1
svc = svm.SVC(kernel='linear', C=1).fit(x, y)
Z = svc.decision_function(np.c_[X.ravel(), Y.ravel()])
Z = Z.reshape(X.shape)

# Plotting the decision boundary for C=1
plt.contourf(X, Y, Z > 0, alpha=0.4)
plt.contour(X, Y, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])
plt.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], s=120, facecolors='none')
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, alpha=0.9)
plt.show()

# Linear SVC with C=0.1
svc = svm.SVC(kernel='linear', C=0.1).fit(x, y)
Z = svc.decision_function(np.c_[X.ravel(), Y.ravel()])
Z = Z.reshape(X.shape)

# Plotting the decision boundary for C=0.1
plt.contourf(X, Y, Z > 0, alpha=0.4)
plt.contour(X, Y, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])
plt.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], s=120, facecolors='none')
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, alpha=0.9)
plt.show()

# Nonlinear SVC *****************************************************************************************

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Data
x = np.array([[1, 3], [1, 2], [1, 1.5], [1.5, 2], [2, 3], [2.5, 1.5],
              [2, 1], [3, 1], [3, 2], [3.5, 1], [3.5, 3]])
y = [0] * 6 + [1] * 5

# SVC with Polynomial Kernel
svc_poly = svm.SVC(kernel='poly', C=1, degree=3).fit(x, y)
X, Y = np.mgrid[0:4:200j, 0:4:200j]
Z = svc_poly.decision_function(np.c_[X.ravel(), Y.ravel()])
Z = Z.reshape(X.shape)

# Plotting decision boundary for Polynomial Kernel
plt.contourf(X, Y, Z > 0, alpha=0.4)
plt.contour(X, Y, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])
plt.scatter(svc_poly.support_vectors_[:, 0], svc_poly.support_vectors_[:, 1], s=120, facecolors='none')
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, alpha=0.9)
plt.show()

# SVC with Radial Basis Function (RBF) Kernel
svc_rbf = svm.SVC(kernel='rbf', C=1, gamma=3).fit(x, y)
Z_rbf = svc_rbf.decision_function(np.c_[X.ravel(), Y.ravel()])
Z_rbf = Z_rbf.reshape(X.shape)

# Plotting decision boundary for RBF Kernel
plt.contourf(X, Y, Z_rbf > 0, alpha=0.4)
plt.contour(X, Y, Z_rbf, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])
plt.scatter(svc_rbf.support_vectors_[:, 0], svc_rbf.support_vectors_[:, 1], s=120, facecolors='none')
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, alpha=0.9)
plt.show()

# Plotting Different SVM Classifiers Using the Iris Dataset ***********************************************


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# Load the Iris dataset
iris = datasets.load_iris()
x = iris.data[:,:2]
y = iris.target

# Linear SVM with Linear Kernel
h = .05
svc = svm.SVC(kernel='linear', C=1.0).fit(x, y)
x_min, x_max = x[:,0].min() - .5, x[:,0].max() + .5
y_min, y_max = x[:,1].min() - .5, x[:,1].max() + .5
h = .02
X, Y = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svc.predict(np.c_[X.ravel(), Y.ravel()])
Z = Z.reshape(X.shape)

# Plot the Linear SVM decision boundaries
plt.contourf(X, Y, Z, alpha=0.4)
plt.contour(X, Y, Z, colors='k')
plt.scatter(x[:,0], x[:,1], c=y)
plt.title("Linear SVM Decision Boundaries")
plt.show()

# Polynomial SVM with degree 3
h = .05
svc = svm.SVC(kernel='poly', C=1.0, degree=3).fit(x, y)
x_min, x_max = x[:,0].min() - .5, x[:,0].max() + .5
y_min, y_max = x[:,1].min() - .5, x[:,1].max() + .5
h = .02
X, Y = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svc.predict(np.c_[X.ravel(), Y.ravel()])
Z = Z.reshape(X.shape)

# Plot the Polynomial SVM decision boundaries
plt.contourf(X, Y, Z, alpha=0.4)
plt.contour(X, Y, Z, colors='k')
plt.scatter(x[:,0], x[:,1], c=y)
plt.title("Polynomial SVM Decision Boundaries (Degree 3)")
plt.show()

# RBF SVM
svc = svm.SVC(kernel='rbf', gamma=3, C=1.0).fit(x, y)

# Plot the RBF SVM decision boundaries
h = .02
X, Y = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svc.predict(np.c_[X.ravel(), Y.ravel()])
Z = Z.reshape(X.shape)

plt.contourf(X, Y, Z, alpha=0.4)
plt.contour(X, Y, Z, colors='k')
plt.scatter(x[:,0], x[:,1], c=y)
plt.title("RBF SVM Decision Boundaries")
plt.show()

# Support Vector Regression (SVR)
diabetes = datasets.load_diabetes()
x_train = diabetes.data[:-20]
y_train = diabetes.target[:-20]
x_test = diabetes.data[-20:]
y_test = diabetes.target[-20:]
x0_test = x_test[:,2]
x0_train = x_train[:,2]
x0_test = x0_test[:,np.newaxis]
x0_train = x0_train[:,np.newaxis]

# Sort the test set and train the SVR models
x0_test.sort(axis=0)
x0_test = x0_test*100
x0_train = x0_train*100

svr = svm.SVR(kernel='linear', C=1000)
svr2 = svm.SVR(kernel='poly', C=1000, degree=2)
svr3 = svm.SVR(kernel='poly', C=1000, degree=3)
svr.fit(x0_train, y_train)
svr2.fit(x0_train, y_train)
svr3.fit(x0_train, y_train)

# Predict and plot the results of SVR
y_pred = svr.predict(x0_test)
y_pred2 = svr2.predict(x0_test)
y_pred3 = svr3.predict(x0_test)

plt.scatter(x0_test, y_test, color='k', label='True Values')
plt.plot(x0_test, y_pred, color='b', label='Linear SVR')
plt.plot(x0_test, y_pred2, color='r', label='Polynomial SVR (Degree 2)')
plt.plot(x0_test, y_pred3, color='g', label='Polynomial SVR (Degree 3)')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.title("Support Vector Regression (SVR)")
plt.show()
