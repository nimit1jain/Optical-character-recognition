import enum
from joblib.memory import FIRST_LINE_TEXT
from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



x,y=fetch_openml('mnist_784',version=1,return_X_y=True)
print("shape of input= ",x.shape,"shape of target= ",y.shape)


# plt.figure()
# for idx,image in enumerate(x[:3]):
#     plt.subplot(1,3,idx+1)
# plt.imshow(np.reshape(image,(28,28)),cmap=plt.cm.gray)
# plt.title('Training: ',y[idx],fontsize=20)

y = [int(i) for i in y] # targets are strings, so need to convert to # int
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=1/7,random_state=0)

model = LogisticRegression(fit_intercept=True,multi_class='auto',penalty='l1',solver='saga',max_iter=1000,C=50,verbose=2,n_jobs=5,tol=0.01)
model.fit(x_train, y_train)


print("Training Accuracy = ", np.around(model.score(x_train,   y_train)*100,3))
print("Testing Accuracy = ", np.around(model.score(x_test, y_test)*100, 3))

from sklearn import metrics
y_test_pred = model.predict(x_test)
cm = metrics.confusion_matrix(y_true=y_test, 
                         y_pred = y_test_pred, 
                        labels = model.classes_)

import seaborn as sns
plt.figure(figsize=(12,12))
sns.heatmap(cm, annot=True, 
            linewidths=.5, square = True, cmap = 'Blues_r', fmt='0.4g')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')