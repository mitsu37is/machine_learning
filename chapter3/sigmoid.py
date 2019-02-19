import numpy as np
import pylab
import seaborn as sns


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cost_1(z):
    return - np.log(sigmoid(z))


def cost_0(z):
    return - np.log(1 - sigmoid(z))


pylab.figure(1)
z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
sns.lineplot(z, phi_z)
pylab.axvline(0, 0, color='k')
pylab.ylim(-0.1, 1.1)
pylab.xlabel('z')
pylab.ylabel('$\phi (z)$')
pylab.yticks([0.0, 0.5, 1.0])
ax = pylab.gca()
ax.yaxis.grid(True)
pylab.tight_layout()
pylab.savefig('images/sigmoid')

pylab.figure(2)
z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)

c1 = [cost_1(x) for x in z]
sns.lineplot(phi_z, c1, label='J(w) if y=1')

c0 = [cost_0(x) for x in z]
sns.lineplot(phi_z, c0, label='J(w) if y=0')

pylab.ylim(0.0, 5.1)
pylab.xlim([0, 1])

pylab.xlabel('$\phi (z)$')
pylab.ylabel('J(w)')
pylab.legend(loc='upper center')

pylab.tight_layout()
pylab.savefig('images/cost')