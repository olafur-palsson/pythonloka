from math import *

import numpy as np
import matplotlib.pyplot as plt

# comment
data2    = np.load('mnist_small.npz')
x_train2 = data2['x_train']
y_train2 = data2['y_train']
x_test2  = data2['x_test']
y_test2  = data2['y_test']
n_train2 = len(y_train2)
n_test2  = len(y_test2)
print(x_train2.shape)
print(x_test2.shape)
print(y_train2.shape)

data1     = np.load('docmatrix.npz')
X         = data1['X']
y         = data1['y']
terms     = data1['terms']
n         = X.shape[0]

rnd       = np.random.permutation(n) # Slembin umrodun talnanna 1,...,n
nfrac     = 0.7 # Hlutfall gagna sem er notad til thjalfunar
n_train1  = int(nfrac*n)
x_train1  = X[rnd[0:n_train1],:]
y_train1  = y[rnd[0:n_train1]]
x_test1   = X[rnd[n_train1:],:]
y_test1   = y[rnd[n_train1:]]

sampleOut = y_train2
sampleIn  = x_train2
testIn    = x_test2
testOut   = y_test2

noClasses = 10 if sampleOut.shape == 10000 else 2

# prenta array til ad debugga
def printa(a):
    for b in a:
        print(b)

# na i vector sem er [1, 0] fyrir retta categoriu annars [0, 1]
def getCategoryVector(category, correctCategory):
    a=np.zeros(2)
    isCorrect = 0 if category == correctCategory else 1
    a[isCorrect] = 1
    return a

def getCategoryMatrix(numberBeingChecked):
    # na i fylki sem er jafn langt og samples
    categoryMatrix = np.zeros((sampleOut.shape[0], noClasses))
    i = 0
    for a in categoryMatrix:
        docCategory = int(y_train2[i]+ 0.001) # passa upp a ad talan se positive
        categoryMatrix[i] = getCategoryVector(docCategory, numberBeingChecked)
        i = i + 1
    return categoryMatrix

# na hnitin a input
def compileCoordinates(numberBeingChecked):
    catMat = getCategoryMatrix(numberBeingChecked)
    return np.matmul(sampleIn.T, catMat)

def columnAverage(a):
    return a.T.dot(np.ones(a.shape[0])) / a.shape[0]

# lengdin fra larettri/lodrettri medaltalslinu
def sumOfSquareDistToAverage(array):
    a = columnAverage(array)
    average = columnAverage(array)
    square = np.vectorize(lambda x : x ** 2)
    return np.sum(square(array - average))

def getMultiplesOfDistancesFromAverage(array):
    averages = columnAverage(array)
    dist = array - averages
    multiplied = np.ones([array.shape[0]])
    i = 0
    for entry in dist:
        multiplied[i] = (entry[0] * entry[1])
        i = i + 1
    return np.sum(multiplied)

def getSlope(numberBeingChecked):
    coordinates = compileCoordinates(numberBeingChecked)
    sumOfMultiples = getMultiplesOfDistancesFromAverage(coordinates)
    sumOfSquareDist = sumOfSquareDistToAverage(coordinates[:,0])
    # sumOfMultiples = np.sum(coordinates)
    return sumOfMultiples / sumOfSquareDist

def leastSquare2(numberBeingChecked):
    slope = getSlope(numberBeingChecked)
    print(slope)
    coordinates = compileCoordinates(numberBeingChecked)
    meanX, meanY = columnAverage(coordinates)
    constant = meanY - slope * meanX
    return slope, constant

def plot(numberBeingChecked):
    coordinates = compileCoordinates(numberBeingChecked)
    slope, constant = leastSquare2(numberBeingChecked)
    plt.plot([0, 500], [constant, constant + 500 * slope])
    plt.scatter(coordinates[:,0], coordinates[:,1])
    plt.show()

def getRotationMatrix(theta):
    return       np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

def getTransformedData(numberBeingChecked):
    # transformar data a thann hatt ad regression linan er larett
    # theta er horn regression linunnar
    slope, constant = leastSquare2(numberBeingChecked)
    theta = -np.arctan(slope)
    transform = getRotationMatrix(theta)
    constantCoordinates = np.array([0, constant])
    # vid tokum herna bara y-gildin post transformation vegna thess ad bara
    # thau hafa ahrif a lengd fra regression linunni
    transformedConstant = np.matmul(transform, constantCoordinates)[1]
    allValues = np.matmul(transform, compileCoordinates(numberBeingChecked).T)[1]
    heightRemoved = allValues - transformedConstant
    return heightRemoved

def getSquaredTransformedValues(keepsigns, numberBeingChecked):
    square = np.vectorize(lambda x : x ** 2)
    squareKeep = np.vectorize( lambda x : -(x ** 2) if x < 0 else x ** 2)
    transformed = getTransformedData(numberBeingChecked)
    squared = squareKeep(transformed) if keepsigns else square(transformed)
    return squared

def get_TermSquareDist_Dictionary(keepSigns, numberBeingChecked = -1):
    squared = getSquaredTransformedValues(keepSigns, numberBeingChecked)
    termvalues = {}
    i = 0
    for term in terms:
        termvalues[term] = squared[i]
        i = i + 1
    return termvalues

def getTop10():
    sortedTermsValues = sorted(termvalues.items(), key=operator.itemgetter(1), reverse=True)
    termvalues = get_TermSquareDist_Dictionary(False, -1)
    for a in range(0, 10):
        print(a)
        print(sortedTermsValues[a])

def getIndexOfBest(array):
    indexOfMax = -1
    maxValue = 0
    for a in range(0, len(array)):
        if array[a] > maxValue:
            indexOfMax = a

def getClassified2(x):
    sampleSize = x.shape[0]
    shapeOfResults = [10, sampleSize, x.shape[1]]
    sampleValueForClass = np.zeros(shapeOfResults)

    # Na i gildin fyrir hvern pixil og hverja tolu
    print("Round 1")
    for i in range(0, 10):
        a = getSquaredTransformedValues(True, i)
        sampleValueForClass[i] = a

    # adeins ad fletja arrayid ut
    print("Round 2")
    print(sampleSize)
    ones = np.ones(x.shape[0])
    print(sampleValueForClass.shape)
    summedUpValues = np.zeros((10, x.shape[1]))
    print(summedUpValues.shape)
    k = 0
    multMinusOne = np.vectorize(lambda x : -x)
    sign = np.vectorize(lambda x : 1 if x > 0 else -1)

    for allValuesForOneNumber in summedUpValues:
        sumOfRows = np.matmul(sampleValueForClass[k].T, ones)
        #sumOfRows = sign(sumOfRows)
        sumOfRows = multMinusOne(sumOfRows)
        summedUpValues[k] = sumOfRows
        k = k + 1

    print("Round 3")
    print(summedUpValues.shape)
    getAbs = np.vectorize(lambda x : abs(x))
    max = np.max(getAbs(summedUpValues))
    divideByMax = np.vectorize(lambda x : round(x/max, 2))
    summedUpValues = divideByMax(summedUpValues)
    i = -1
    classified = np.array([])
    for sample in x:
        i = i + 1
        maxValue = -10000000000000000000000000
        indexMax = -1
        for j in range(0, summedUpValues.shape[0]):
            value = sample.dot(summedUpValues[j])
            if value > maxValue:
                maxValue = value
                indexMax = j
        print(np.array([i, sampleOut[i], indexMax, maxValue]).astype(int))
        classified = np.append(classified, indexMax)

    return classified

yGuesses = getClassified2(sampleIn)

confusionMatrix = np.zeros((10, 10))
correct = 0
i = -1
for y in sampleOut:
    i = i + 1
    confusionMatrix[int(y)][int(yGuesses[i] + 0.01)] = confusionMatrix[int(y)][int(yGuesses[i])] + 1
    if int(y) == yGuesses[i]:
        correct = correct + 1
        continue

zero = 0
notZero = 0
for guess in yGuesses:
    if guess == 0:
        zero = zero + 1

print(float(zero) / yGuesses.shape[0])
print("Confusion Matrix")

print(confusionMatrix.astype(int))
print()
print("Success Rate")
print(float(correct) / float(sampleOut.shape[0]))

plot(5)
