import pylab
import seaborn as sns
import numpy as np
import pandas as pd
import adaline_sgd
import decision_regions

sns.set(style="whitegrid")
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1 ,1)
X = df.iloc[0:100, [0, 2]].values

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

pylab.figure(1)
ada = adaline_sgd.AdalineSGD(n_iter=15, eta=0.01, random_state=1).fit(X_std, y)
decision_regions.plot_decision_regions(X_std, y, classifier=ada)
pylab.title('Adaline - Stochastic Gradient Descent')
pylab.xlabel('sepal length [standardized]')
pylab.ylabel('petal length [standardized]')
pylab.legend(loc='upper left')
pylab.tight_layout()
pylab.savefig('chapter2/images/adaline_sgd_decision')

pylab.figure(2)
pylab.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
pylab.xlabel('Epochs')
pylab.ylabel('Average Cost')
pylab.savefig('chapter2/images/standard_adaline_sgd_cost')