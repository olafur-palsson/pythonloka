

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

'''
blablabla
'''

def getCategoryVector(category):
    return np.array([[1, 0]]) if category == 1 else np.array([[0, 1]])

# skilar 600x3
def getCategoryMatrix():
    categoryMatrix = np.zeros((600, 2))
    i = 0
    for docCategory in y:
        docCategory = int(docCategory + 0.25)
        categoryMatrix[i] = getCategoryVector(docCategory)
        i = i + 1

    return categoryMatrix

# skilar 1000x3
def complileCoordinates():
    catMat = getCategoryMatrix()
    blablabla = np.matmul(X.T, catMat)
    return blablabla

def get_sign():
    sign = lambda cat : 1 if cat == 1 else -1
    sign = np.vectorize(sign)
    return sign(y)

def get_b():
    return np.matmul(X.T, get_sign())

def leastSquare():
    coordinates = complileCoordinates()
    b = get_b()
    return np.linalg.lstsq(coordinates, get_b())

a = leastSquare()
for b in a:
    print(b)
print("done")
