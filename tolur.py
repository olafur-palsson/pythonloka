
from math import *
import numpy as np
import matplotlib.pyplot as plt
import operator

# comment
data2    = np.load('mnist_small.npz')
x_train2 = data2['x_train']
y_train2 = data2['y_train']
x_test2  = data2['x_test']
y_test2  = data2['y_test']
n_train2 = len(y_train2)
n_test2  = len(y_test2)

data1     = np.load('docmatrix.npz')
X         = data1['X']
y         = data1['y']
terms     = data1['terms']
n         = X.shape[0]

toBinary = np.vectorize(lambda x : 1 if x== 1 else 0)
y = toBinary(y)

rnd       = np.random.permutation(n) # Slembin umrodun talnanna 1,...,n
nfrac     = 0.7 # Hlutfall gagna sem er notad til thjalfunar
n_train1  = int(nfrac*n)
x_train1  = X[rnd[0:n_train1],:]
y_train1  = y[rnd[0:n_train1]]
x_test1   = X[rnd[n_train1:],:]
y_test1   = y[rnd[n_train1:]]

print(y_test1, y_train1)

sampleOut = y_train2
sampleIn  = x_train2
testIn    = x_test2
testOut   = y_test2

#stokka aftur til ad fa betra training session
def shuffle():
    global x_train1, y_train1, x_test1, y_test1
    rnd       = np.random.permutation(n) # Slembin umrodun talnanna 1,...,n
    x_train1  = X[rnd[0:n_train1],:]
    y_train1  = y[rnd[0:n_train1]]
    x_test1   = X[rnd[n_train1:],:]
    y_test1   = y[rnd[n_train1:]]

# her veljum vid hvor datasettid vid aetlum ad nota
def setSamples(isMNIST):
    global sampleIn, sampleOut, testIn, testOut
    if isMNIST:
        sampleOut = y_train2
        sampleIn = x_train2
        testOut = y_test2
        testIn = x_test2
    else:
        sampleOut = y_train1
        sampleIn = x_train1
        testOut = y_test1
        testIn = x_test1

# skilar hversu mikif af flokkum
def getNoClasses():
    return 10 if sampleOut.shape[0] == 10000 else 2

# prenta array til ad debugga/skoda data
def printa(a):
    for b in a:
        print(b)

# na i vector sem er [1, 0] fyrir retta categoriu annars [0, 1]
def getCategoryVector(category, correctCategory):
    a=np.zeros(2)
    isCorrect = 0 if category == correctCategory else 1
    a[isCorrect] = 1
    return a

# na i fylki af [0, 1]/[1, 0] eftir thvi hvada tolu vid erum ad skoda
def getCategoryMatrix(numberBeingChecked):
    # na i fylki sem er jafn langt og samples
    categoryMatrix = np.zeros((sampleOut.shape[0], 2))
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

# medaltal dalka
def columnAverage(a):
    averages = np.matmul(a.T, np.ones(a.shape[0])) / a.shape[0]
    return averages

# lengdin fra larettri/lodrettri medaltalslinu
def sumOfSquareDistToAverage(array):
    a = columnAverage(array)
    average = columnAverage(array)
    square = np.vectorize(lambda x : x ** 2)
    return np.sum(square(array - average))

# na i fjarlaegdina fra medaltali fyrir regression linu
def getMultiplesOfDistancesFromAverage(array):
    averages = columnAverage(array)
    dist = array - averages
    multiplied = np.ones([array.shape[0]])
    i = 0
    for entry in dist:
        multiplied[i] = (entry[0] * entry[1])
        i = i + 1
    return np.sum(multiplied)

# na i hallatolu regression linu
def getSlope(numberBeingChecked):
    coordinates = compileCoordinates(numberBeingChecked)
    sumOfMultiples = getMultiplesOfDistancesFromAverage(coordinates)
    sumOfSquareDist = sumOfSquareDistToAverage(coordinates[:,0])
    #sumOfMultiples = np.sum(coordinates)
    return sumOfMultiples / sumOfSquareDist

# na i hallatolu og fastann til ad klara regression linuna
def leastSquare2(numberBeingChecked):
    slope = getSlope(numberBeingChecked)
    coordinates = compileCoordinates(numberBeingChecked)
    meanX, meanY = columnAverage(coordinates)
    constant = meanY - slope * meanX
    return slope, constant

# thaegilegt til ad plotta upp binary classifier
def plot(numberBeingChecked):
    coordinates = compileCoordinates(numberBeingChecked)
    slope, constant = leastSquare2(numberBeingChecked)
    plt.plot([0, 500], [constant, constant + 500 * slope])
    plt.scatter(coordinates[:,0], coordinates[:,1])
    plt.show()

# skilar fylki sem snyr gognum hornid theta
def getRotationMatrix(theta):
    return       np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

# skilar gognunum snunum thannig ad regression linan er larett
# vid skilum sidan y hnitunum sem verda tha fjarlaegd fra linunni
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

# herna skilad square distance fra regression linu, vid getum sidan haldid
# formerkinu
def getSquaredTransformedValues(keepsigns, numberBeingChecked):
    square = np.vectorize(lambda x : x ** 2)
    squareKeep = np.vectorize( lambda x : -(x ** 2) if x < 0 else x ** 2)
    transformed = getTransformedData(numberBeingChecked)
    squared = squareKeep(transformed) if keepsigns else square(transformed)
    squared = transformed
    return squared

# fyrir ordalistann, na i ordalistann og gefa hverju ordi gildi
# sem er fjarlaegd fra regression linu
def get_TermSquareDist_Dictionary(keepSigns, numberBeingChecked = -1):
    squared = getSquaredTransformedValues(keepSigns, numberBeingChecked)
    termvalues = {}
    i = 0
    for term in terms:
        termvalues[term] = squared[i]
        i = i + 1
    return termvalues

# radar ordunum eftir hversu langt thau eru fra linunni
# skilar 10 eftstu
def getTop10():
    termvalues = get_TermSquareDist_Dictionary(False, 1)
    sortedTermsValues = sorted(termvalues.items(), key=operator.itemgetter(1), reverse=True)
    for a in range(0, 10):
        print(a, sortedTermsValues[a][0].split("\'"))

# skilar index af haesta gildi i vigri
def getIndexOfBest(array):
    indexOfMax = -1
    maxValue = 0
    for a in range(0, len(array)):
        if array[a] > maxValue:
            indexOfMax = a

# tekur in vigur og deilir med haesta stakinu vigrinum
# til ad fa tolu a bilinu [-1: 1]
def normalize(a):
    getAbs = np.vectorize(lambda x : abs(x))
    max = np.max(getAbs(a))
    divideByMax = np.vectorize(lambda x : round(x/max, 3))
    return divideByMax(a)

# least square classifier, tekur in gogn og skilar agiskunum
def getClassified2(x):
    sampleSize = x.shape[0]
    shapeOfResults = [getNoClasses(), sampleSize, x.shape[1]]
    sampleValueForClass = np.zeros(shapeOfResults)

    # Na i gildin fyrir hvern pixil og hverja tolu
    for i in range(0, getNoClasses()):
        a = getSquaredTransformedValues(True, i)
        sampleValueForClass[i] = a

    # notad til ad fa medaltal
    ones = np.ones(x.shape[0])

    # nokkur hjalparfoll
    multMinusOne = np.vectorize(lambda x : -x)
    sign = np.vectorize(lambda x : 1 if x > 0 else -1)

    # naum i hvada gildi hver pixel hefur fyrir hverja tolu
    summedUpValues = np.zeros((getNoClasses(), x.shape[1]))
    for k in range(0, summedUpValues.shape[0]):
        sumOfRows = columnAverage(sampleValueForClass[k])
        #sumOfRows = sign(sumOfRows)
        sumOfRows = multMinusOne(sumOfRows)
        sumOfRows = normalize(sumOfRows)
        summedUpValues[k] = sumOfRows

    # her flokkum vid gognin
    classified = np.array([])
    for i in range(0, x.shape[0]):
        sample = x[i]
        maxValue = -10000000000000000000000000
        indexMax = -1
        for j in range(0, summedUpValues.shape[0]):
            value = sample.dot(summedUpValues[j])
            if value > maxValue:
                maxValue = value
                indexMax = j
        classified = np.append(classified, indexMax)
    return classified

# litid fall sem telur nidurstodurnar og setur thaer upp
def getConfusionMatrix(guesses, actualOutput):
    confusionMatrix = np.zeros((getNoClasses(), getNoClasses()))
    correct = 0
    i = -1
    for actual in actualOutput:
        i = i + 1
        guess = int(guesses[i])
        y     = int(actual)
        confusionMatrix[y][guess] = confusionMatrix[y][guess] + 1
        if y == guess:
            correct = correct + 1
            continue
    successRate = float(correct) / guesses.shape[0]
    return successRate, confusionMatrix.astype(int)

# litid fall sem ser um ad prenta nidurstodurnar
def printConfusion(guesses, actual):
    success, confusion = getConfusionMatrix(guesses, actual)
    print("Error Rate:")
    print(1 - success)
    print("Confusion Matrix")
    print(confusion)
    print("")

# herna er er fall sem velur hvort vid aetlum ad profa
# ordalistan eda MNIST
def printConfusionAndSuccessRate(isMNIST):
    setSamples(isMNIST)
    print(sampleIn.shape)
    trainResults = getClassified2(sampleIn)
    testResults  = getClassified2(testIn)
    print("Training Data")
    printConfusion(trainResults, sampleOut)
    print("Test Data")
    printConfusion(testResults, testOut)

# brute force stokka thangad til nidurstadan flokkar vel
def trialAndError():
    print("trial and error")
    success = 1
    confusion = 0
    trial = 0
    print("(This will take some time)")
    while success > 0.1:
        trial = trial + 1
        shuffle()
        setSamples(False)
        trainResults = getClassified2(sampleIn)
        success, confusion = getConfusionMatrix(trainResults, sampleOut)
    printConfusionAndSuccessRate(False)
    print("done")
cachedSampleIn = np.copy(sampleIn)
cachedTestIn = testIn

# vid buum her til synthetic features
def engineerFeatures(howManyFeatures):
    global x_train2, x_test2, cachedTestIn, cachedSampleIn
    max = np.max(cachedSampleIn)
    newSampleIn = np.zeros((cachedSampleIn.shape[0], cachedSampleIn.shape[1] + howManyFeatures))
    newTestIn   = np.zeros((cachedTestIn.shape[0], cachedTestIn.shape[1] + howManyFeatures))
    sign = np.vectorize(lambda x : -1 if x<0 else 1)
    randomWeights = np.random.rand(cachedSampleIn.shape[1], howManyFeatures)
    #randomWeights = sign(randomWeights)
    trainFeatures = np.matmul(cachedSampleIn, randomWeights) / 784
    testFeatures = np.matmul(cachedTestIn, randomWeights) / 784
    for i in range(0, cachedSampleIn.shape[0]):
        newSampleIn[i] = np.append(cachedSampleIn[i], trainFeatures[i])
    for i in range(0, cachedTestIn.shape[0]):
        newTestIn[i] = np.append(cachedTestIn[i], testFeatures[i])
    x_train2 = newSampleIn
    x_test2 = newTestIn

print("----------TERM Classification-----------")
trialAndError()
getTop10()
print("----------MNIST Classification-----------")
printConfusionAndSuccessRate(True)
engineerFeatures(100)
print("----------MNIST extra features-----------")
printConfusionAndSuccessRate(True)
engineerFeatures(1000)
printConfusionAndSuccessRate(True)
engineerFeatures(5000)
printConfusionAndSuccessRate(True)
engineerFeatures(10000)
printConfusionAndSuccessRate(True)
