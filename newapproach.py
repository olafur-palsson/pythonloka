

import numpy as np

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
    return {
        1: np.array([1, 0]),
        2: np.array([0, 1]),
        3: np.array([0, 1])
    }[category]

# skilar 600x3
def getCategoryMatrix():
    categoryMatrix = np.array([])
    for docCategory in y:
        np.append(categoryMatrix, getCategoryVector(docCategory))

# skilar 1000x3
def complileCoordinates():
    catMat = getCategoryMatrix()
    return np.matmul(X.T, catMat)

coordinates = complileCoordinates()

for c in coordinates:
    print(c)
