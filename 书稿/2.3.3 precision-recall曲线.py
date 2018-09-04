# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import numpy as np

X,y = datasets.load_iris(return_X_y=True)
# Add noisy features
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape

X = np.c_[X, random_state.randn(n_samples, 200*n_features)]
# Limit to the two first classes, and split into training and test
X_train, X_test, y_train, y_test = train_test_split(X[y < 2], y[y < 2],
                                                    test_size=.5,random_state=random_state)
# Create a simple classifier
classifier = svm.LinearSVC(random_state=random_state)
classifier.fit(X_train, y_train)
y_score = classifier.decision_function(X_test)
y_pred = classifier.predict(X_test)

# Compute the average precision score
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))

# Plot the Precision-Recall curve
from sklearn.metrics import precision_recall_curve,f1_score
import matplotlib.pyplot as plt
precision, recall, _ = precision_recall_curve(y_test, y_score)
f1 = 2 * precision * recall / ( precision + recall )
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
plt.plot(recall,f1,'--')
plt.xlabel('查全率-Recall',fontproperties=myfont)
plt.ylabel('查准率-Precision',fontproperties=myfont)
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('“查准率-查全率”曲线: AUC={0:0.2f}'.format(average_precision),
          fontproperties=myfont)
plt.show()