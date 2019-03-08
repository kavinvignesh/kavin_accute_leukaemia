from numpy import genfromtxt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import pylab as plt
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import array
import csv
#loading and pruning the data
dataset = genfromtxt('dataKMWS.csv', dtype = float, delimiter = ',')
x = dataset[:,0:2] #Feature set
y = dataset[:,3] #Label set
target_names = ['0','1']
#method to plot the graph for reduced dimensions
def plot_2D(data, target, target_names):
	colors = cycle('rgbcmykw')
	target_ids = range(len(target_names))
	plt.figure()
	for i, c, label in zip(target_ids, colors, target_names):
		plt.scatter(data[target == i, 0],data[target ==i, 1], c = c, label = label)
		plt.legend()
# Classifying the data using a Linear SVM and predicting the probability of disease belonging to a particular class
modelSVM = LinearSVC(C = 0.1)
pca = PCA(n_components = 2, whiten = True).fit(x)
x_new = pca.transform(x)
#calling plot_2D
plot_2D(x_new, y, target_names)
plt.show()	
#Applying cross validation on the training and test set for validating linear SVM model
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.2, train_size = 0.8, random_state = 0)
modelSVM = modelSVM.fit(x_train, y_train)
print ("Linear SVC Accuracy: "+str(modelSVM.score(x_test, y_test)))
modelSVMRaw = LinearSVC(C=0.1)
modelSVMRaw = modelSVMRaw.fit(x_new, y)
cnt = 0
s = 0
for i in modelSVMRaw.predict(x_new):
	if y[s] == 0:
		cnt = cnt + 1
	s = s+1
#Applying the principal component analysis on the data features
modelSVM2 = SVC(C = 0.1, kernel = 'rbf')
#Applying cross validation on the training and test set for validating our linear svm model
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_new, y, test_size=0.2, train_size=0.8, random_state=0)
modelSVM2 = modelSVM2.fit(x_train1, y_train1)
print ("RBF SVC Accuracy: "+str(modelSVM2.score(x_test1, y_test1)))
modelSVMRaw = SVC(C=0.1, kernel = 'rbf')
modelSVMRaw = modelSVMRaw.fit(x_new, y)
cnt1 = 0
s = 0
for i in modelSVMRaw.predict(x_new):
	if y[s] == 0:
		cnt1 = cnt1 + 1
	s = s+1
dataset = genfromtxt('positive.csv', dtype = float, delimiter = ',')
q = dataset[0:3] #Feature set
a = dataset[0:2]
print ("Test data: "+str(q))
#b = dataset[:,3] #Label set
ax = a.reshape(1, -1)
svm = SVC(kernel='linear').fit(x_train, y_train)
rbf_svc = SVC( gamma = 0.7).fit(x_train, y_train)
p = PCA(n_components = 2, whiten = True).fit(x)
a_new = p.transform(ax)
print ("PCA transformation in the test data: "+str(a_new))
modelSVMRaw = LinearSVC(C=0.1)
modelSVMRaw = modelSVMRaw.fit(x_new, y)
b = modelSVMRaw.predict(a_new)
e = int(b)
print("Prediction: "+str(e))
if(e == 1):
	print ("Result : Cancerous")
else:
	print ("Result : Non - Cancerous")
#create a mesh to plot in
x_min, x_max = x_new[:, 0].min()- 1, x_new[:, 0].max()+ 1
y_min, y_max = x_new[:, 1].min()- 1, x_new[:, 1].max()+ 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
#title for the plots
titles = ['SVC with linear kernal', 'SVC with RBF kernal']
#plot the decision boundary 
for i,modelSVM2 in enumerate((svm, rbf_svc)):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace = 0.4, hspace = 0.4)
    z = modelSVM2.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap = plt.cm.Paired, alpha = 0.8)
    plt.scatter(x_new[:, 0], x_new[:, 1], c = y, cmap = plt.cm.Paired)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks()
    plt.yticks()
    plt.title(titles[i])
plt.show()
b = int(b)
# writing test data to training data set file
with open('prediction_dataset.csv', 'a') as csvfile:
		filewriter = csv.writer(csvfile, delimiter=',', lineterminator = '\n', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		filewriter.writerow([q[0], q[1], q[2], b])
