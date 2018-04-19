

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

def printa(a):
    for b in a:
        print(b)

def addOnes(coordinates):
    result = np.ones((1000, 3))
    print(coordinates[0])
    print(result[0])
    i = 0
    j = 0
    while i < 1000:
        while j < 1:
            result[i][j] = coordinates[i][j]
            j = j + 1
        i = i + 1
    return result

def columnAverage(a):
    return a.T.dot(np.ones(a.shape[0])) / a.shape[0]

def sumOfSquareDistToAverage(array):
    a = columnAverage(array)
    print(a)
    average = columnAverage(array)
    square = np.vectorize(lambda x : x ** 2)
    return np.sum(square(array - average))

def getMultiplesOfDistancesFromAverage(array):
    averages = columnAverage(array)
    dist = array - averages
    multiplied = np.ones([array.shape[0]])
    i = 0
    for entry in dist:
        print(entry[0])
        print(entry[1])
        multiplied[i] = (entry[0] * entry[1])
        i = i + 1
    return np.sum(multiplied)

def getSlope():
    coordinates = complileCoordinates()
    sumOfMultiples = getMultiplesOfDistancesFromAverage(coordinates)
    sumOfSquareDist = sumOfSquareDistToAverage(coordinates[:,0])
    print("this thang")
    print(sumOfMultiples)
    print(sumOfSquareDist)
    return sumOfMultiples / sumOfSquareDist

def leastSquare2():
    slope = getSlope()
    coordinates = complileCoordinates()
    meanX, meanY = columnAverage(coordinates)
    constant = meanY - slope * meanX
    return slope, constant

def leastSquare():
    coordinates = complileCoordinates()
    #coordinates = addOnes(coordinates)
    b = get_sign()
    return np.linalg.lstsq(coordinates, get_b())

def plot():
    coordinates = complileCoordinates()
    plot(coordinates[:,0], coordinates[:,1])


print(leastSquare2())
plot()

a = leastSquare()
printa(a)
print("done")
