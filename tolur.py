import numpy as np

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
def complileCoordinates(numberBeingChecked):
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
    coordinates = complileCoordinates(numberBeingChecked)
    sumOfMultiples = getMultiplesOfDistancesFromAverage(coordinates)
    sumOfSquareDist = sumOfSquareDistToAverage(coordinates[:,0])
    return sumOfMultiples / sumOfSquareDist

def leastSquare2(numberBeingChecked):
    slope = getSlope(numberBeingChecked)
    coordinates = complileCoordinates(numberBeingChecked)
    arrayOfAverages = columnAverage(coordinates)
    constant = meanY - slope * meanX
    return slope, constant

def plot(numberBeingChecked):
    coordinates = complileCoordinates()
    slope, constant = leastSquare2(numberBeingChecked)
    plt.plot([0, 500], [constant, constant + 500 * slope])
    plt.scatter(coordinates[:,0], coordinates[:,1])
    plt.show()

def getTransformedData(numberBeingChecked):
    # transformar data a thann hatt ad regression linan er larett
    # theta er horn regression linunnar
    theta = -np.arctan(getSlope(numberBeingChecked))
    transform = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta) ]])
    return np.matmul(transform, complileCoordinates(numberBeingChecked).T)

def getSquaredTransformedValues(keepsigns, numberBeingChecked):
    square = np.vectorize(lambda x : abs(x))
    squareKeep = np.vectorize( lambda x : -(x ** 2) if x > 0 else x ** 2)
    transformed = getTransformedData(numberBeingChecked)
    squared = squareKeep(transformed) if keepsigns else square(transformed)
    return squared

def get_TermSquareDist_Dictionary(keepSigns, numberBeingChecked):
    squared = getSquaredTransformedValues(keepSigns, numberBeingChecked)
    termvalues = {}
    i = 0
    for term in terms:
        termvalues[term] = squared[1][i]
        i = i + 1
    return termvalues

termvalues = get_TermSquareDist_Dictionary(False, 1)
termValuesWithSigns = (getSquaredTransformedValues(True, 1))[1]
sortedTermsValues = sorted(termvalues.items(), key=operator.itemgetter(1), reverse=True)

print(termvalues.shape)

'''
for a in range(0, 10):
    print(a)
    print(sortedTermsValues[a])
'''

def getClassified(x):
    sign = np.vectorize(lambda x : 1 if x > 0 else -1)
    values = np.matmul(x, termValuesWithSigns.T)
    return sign(values)

yGuesses = getClassified(x_test)

correct = 0
i = -1
for y in y_test:
    i = i + 1
    if int(y) == yGuesses[i]:
        correct = correct + 1
        continue
    if int(y) != 1 and yGuesses[i] == -1:
        correct = correct + 1
        continue

print(1-(float(correct) / float(y_test.shape[0])))

plot()
