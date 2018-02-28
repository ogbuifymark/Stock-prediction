__author__ = 'DELL'

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import time
import functools


totalStart = time.time()
def bytedate2num(fmt):
    def converter(B):
        return mdates.strpdate2num(fmt)(B.decode('ascii'))
    return converter


date_converter = bytedate2num("%Y%m%d%H%M%S")
date, bid, ask = np.loadtxt('GBPUSD1d.txt', unpack=True, delimiter=',', converters={0: date_converter})

def percentChange(startPoint, currentPoint):
    try:
        x = ((float(currentPoint) - startPoint) / abs(startPoint)) * 100.00
        if x == 0.0:
            return 0.00000000001
        else:
            return x
    except:
        return 0.00000000001


def patternStorage():
    patStartTime = time.time()
    x = len(avgLine) - 60
    y = 31

    while y < x:
        pattern = []
        p = 0
        while p < 30:
            pattern.append(percentChange(avgLine[y - 30], avgLine[(y - 1)-(p+1)]))
            p +=1
        outcomeRange = avgLine[y + 20:y + 30]
        currentPoint = avgLine[y]
        try:
            avgOutcome = (functools.reduce(lambda x, y: x + y, outcomeRange) / len(outcomeRange))
        except Exception as e:
            print(str(e))
            avgOutcome =0
        futureOutcome = percentChange(currentPoint, avgOutcome)

        patternArr.append(pattern)
        performanceArr.append(futureOutcome)

        y += 1
    patEndTime = time.time()
    print(len(patternArr))
    print(len(performanceArr))
    print("Pattern storage took:", patEndTime - patStartTime, 'seconds')

def currentPattern():
    current =0
    while current < 30:
        patForRec.append(percentChange(avgLine[-31], avgLine[-(31 - (current+1))]))
        current +=1
    print(patForRec)
def patternRecognition():
    predictedOutcomeArr = []
    patFound = 0
    patFoundArr = []

    for eachPattern in patternArr:
        howSim = 0
        sim =0
        while sim < 30:
            similarity = 100.00 - abs(percentChange(eachPattern[sim], patForRec[sim]))

            howSim += similarity
            sim += 1

        howSim = howSim/30.00

        if howSim > 70:
            patindex = patternArr.index(eachPattern)
            patFound = 1
            '''print('##############')
            print('##############')
            print(patForRec)
            print('===============')
            print
            (eachPattern)
            print("------------------")
            print("predicted outcome", performanceArr[patindex])'''


            xp = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
            patFoundArr.append(eachPattern)

    predAray = []
    if patFound == 1:
        #fig = plt.figure(figsize=(10,6))
        for eachpath in patFoundArr:
            futurePoints = patFoundArr.index(eachpath)

            if performanceArr[futurePoints] > patForRec[29]:
                pcolor = '#24bc00'

                predAray.append(1.000)
            else:
                pcolor = '#d40000'
                predAray.append(-1.000)

            #plt.plot(xp, eachpath)
            predictedOutcomeArr.append(performanceArr[futurePoints])
            #plt.scatter(35,performanceArr[futurePoints], c=pcolor, alpha=.3)

        realOutcomeRange = allData[toWhat+20:toWhat+30]
        realAvgOutcome = (functools.reduce(lambda x, y: x + y, realOutcomeRange) / len(realOutcomeRange))
        realMovement = percentChange(allData[toWhat], realAvgOutcome)
        predictedAvgOutcome = (functools.reduce(lambda x, y: x + y, predictedOutcomeArr) / len(predictedOutcomeArr))


        print(predAray)
        predictionAverage = functools.reduce(lambda  x, y: x+y, predAray)/ len(predAray)

        print(predictionAverage)
        if predictionAverage < 0:
            print('drop predicted')
            print(patForRec[29])
            print(realMovement)
            if realMovement < patForRec[29]:
                accuracyArr.append(100)
            else:
                accuracyArr.append(0)

        if predictionAverage >0:
            print('rise predicted')
            print(patForRec[29])
            print(realMovement)
            if realMovement > patForRec[29]:
                accuracyArr.append(100)
            else:
                accuracyArr.append(0)
        #plt.scatter(40, realMovement, c= '#54fff7', s=25)
        #plt.scatter(40, predictedAvgOutcome, c='b', a=25)


        #plt.plot(xp, patForRec,'#54fff7', linewidth = 3)
        #plt.grid(True)
        #plt.title('Pattern Recognition')
        #plt.show()


def graphRawFx():
    fig = plt.figure(figsize=(10, 7))
    ax1 = plt.subplot2grid((40, 40), (0, 0), rowspan=40, colspan=40)

    ax1.plot(date, bid)
    ax1.plot(date, ask)
    plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)

    ax1_2 = ax1.twinx()
    ax1_2.fill_between(date, 0, (ask - bid), facecolors='g', alpha=.3)
    plt.subplots_adjust(bottom=.23)

    plt.grid(True)
    plt.show()


# graphRawFx()
datalength = int(bid.shape[0])
print('data length is ', datalength)

toWhat = 37000
allData = ((bid * ask) / 2)

accuracyArr = []
samps = 0

while toWhat < datalength:
    # avgLine = ((bid * ask) / 2)
    avgLine = allData[:toWhat]
    patternArr = []
    performanceArr = []
    patForRec = []

    patternStorage()
    currentPattern()

    patternRecognition()
    totalTime = time.time() - totalStart
    print('Entire processing time = ', totalTime, 'seconds')

    samps +=1
    toWhat += 1
    accuracyAverage = (functools.reduce(lambda x, y: x + y, accuracyArr) / len(accuracyArr))
    print('Backtested Accuracy is ', str(accuracyAverage)+ '% after', samps,'samples' )
