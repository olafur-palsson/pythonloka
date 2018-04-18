import numpy as np
import sys
# sys.setdefaultencoding() does not exist, here!
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')

data=np.load('docmatrix.npz')
X=data['X']
y=data['y']
terms=data['terms']

# Skipta i gognum i thjalfunar og profunargogn
n=X.shape[0]
rnd=np.random.permutation(n) # Slembin umrodun talnanna 1,...,n
nfrac=0.7 # Hlutfall gagna sem er notad til thjalfunar
n_train=int(nfrac*n)
x_train=X[rnd[0:n_train],:]
y_train=y[rnd[0:n_train]]
x_test=X[rnd[n_train:],:]
y_test=y[rnd[n_train:]]

def getCategoryVector(category):
    f = np.vectorize(lambda x: 1 if x == category else -1)
    return f(y);


def aHat():





for line in data['X']:
    print(terms.shape)

print(X.shape)
print(y.shape)
print(terms.shape)

print(data['y'])

print(data)
print("X_train:", x_train.shape)
print("X_test:", x_test.shape)
