import pandas as pd
import pylab
import seaborn as sns
import numpy as np
import perceptron
import decision_regions

sns.set(style="whitegrid")
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

pylab.figure(1)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1 ,1)
X = df.iloc[0:100, [0, 2]].values
sns.scatterplot(X[:50, 0], X[:50, 1], label="setosa")
sns.scatterplot(X[50:100, 0], X[50:100, 1], label="versicolor")
pylab.xlabel('sepal length [cm]')
pylab.ylabel('petal length [cm]')
pylab.legend(loc='upper left')
pylab.savefig('chapter2/images/data_scatter')

pylab.figure(2)
ppn = perceptron.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
sns.lineplot(range(1, len(ppn.errors_) + 1), ppn.errors_)
pylab.xlabel('Epochs')
pylab.ylabel('Number of updates')
pylab.savefig('chapter2/images/errors')

pylab.figure(3)
decision_regions.plot_decision_regions(X, y, classifier=ppn)
pylab.xlabel('sepal length [cm]')
pylab.ylabel('petal length [cm]')
pylab.legend(loc='upper left')
pylab.savefig('chapter2/images/decision_regions')