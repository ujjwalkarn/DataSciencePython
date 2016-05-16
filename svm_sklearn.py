import numpy as np
from sklearn import svm

# Read the data
train = np.loadtxt(open("train.csv","rb"), delimiter=",", skiprows=0)
trainLabels = np.loadtxt(open("trainLabels.csv","rb"), delimiter=",", skiprows=0)
test = np.loadtxt(open("test.csv","rb"), delimiter=",", skiprows=0)


X, y = train, trainLabels
s = svm.SVC()
s.fit(X, y)

predictions = s.predict(test)
np.savetxt("fancySVMSubmission.csv", predictions.astype(int), fmt='%d', delimiter=",")
