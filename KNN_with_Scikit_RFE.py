import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import time
import speechpy
import scipy.io.wavfile as wav
from sympy.ntheory import factorint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def featuresToText(features):
    feats = open('features.txt', 'w')
    feats.write(features)

def convertTo1Dim(square):
    array = [0]*len(square)*len(square[0])

    for i in range(len(square)):
        for j in range(len(square[0])):
            
            print square[i][j], i*len(square) + j
            array[i*len(square[0]) + j] = square[i][j]

    return array

#just set irrevs to 0?
def cuttingTheFat(features, fit):
    for i in range(len(features)):
        bestFeatures = fit.support_
        feature = []
        for j in range(len(features[i])):
            if not bestFeatures[j]:
                features[i][j] = ''

        features[i] = filter(lambda a: a != '', features[i])

def cuttingTheFat2(feature, fit):
        bestFeatures = fit.support_
        newFeature = []
        for j in range(len(feature)):
            if bestFeatures[j]:
                newFeature.append(feature[j])

        return newFeature
                

def normalizeFeatures(f):
    f = np.array(f)
    
    a = f / f.__abs__().max(0).astype(float)
    a = (a + 1) / 2.0
    return a

def knn(k, vector, features):
    distances = []
    for i in range(11):
        for j in range(len(features[i])):
            distance = distanceFunction(vector, features[i][j][0])
            distances.append((distance, i))

    sortedDistances = sorted(distances)
    #print distances #holy shit that's a lot

    #get knn
    count = [0]*11
    for i in range(k):
        inst = sortedDistances[i][1]
        count[inst] = count[inst] + 1

    maxi = count[0]
    inst = 0
    for i in range(1, 11):
        if count[i] > maxi:
            inst = i
            maxi = count[i]

    """
    if inst != 2:
        print "Wrong instrument"
    """
            
    return inst
        

def distanceFunction(vector1, vector2):
    distance = 0.0
    for i in range(len(vector1)):
        distance = distance + np.square(vector1[i]-vector2[i]) 
        
    return distance

def arrayAverage(array):
    result = array[0]

    for i in range(1, len(array)):
        result = result + array[i]
    result = result / len(array)

    return result

FACTOR_LIMIT = 5
BEST_LENGTHS = {}

def bestFFTlength(n):
    n_start = n
    n_end = n
    if n_start not in BEST_LENGTHS:
        while max(factorint(n_end)) >= FACTOR_LIMIT:
            n_end -= 1
            BEST_LENGTHS[n_start] = n_end
        else:
            n_end = BEST_LENGTHS[n]
            # n_end
    return n_end

cel = os.getcwd() + '\\IRMAS-TrainingData\\cel'
cla = os.getcwd() + '\\IRMAS-TrainingData\\cla'
flu = os.getcwd() + '\\IRMAS-TrainingData\\flu'
gac = os.getcwd() + '\\IRMAS-TrainingData\\gac'
gel = os.getcwd() + '\\IRMAS-TrainingData\\gel'
org = os.getcwd() + '\\IRMAS-TrainingData\\org'
pia = os.getcwd() + '\\IRMAS-TrainingData\\pia'
sax = os.getcwd() + '\\IRMAS-TrainingData\\sax'
tru = os.getcwd() + '\\IRMAS-TrainingData\\tru'
vio = os.getcwd() + '\\IRMAS-TrainingData\\vio'
voi = os.getcwd() + '\\IRMAS-TrainingData\\voi'
folders = [cel, cla, flu, gac, gel, org, pia, sax, tru, vio, voi]
instruments = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
features = []
classifications = []

#feature extraction

start = time.time()
for i in range(len(folders)):
    j = 0
    
    for filename in os.listdir(folders[i]):
        (fs, signal) = wav.read(os.path.join(folders[i], filename))
        signal = signal[:,0]
        #test different features latter
        fft_length = 131072
        frame_length = fft_length/fs
        
        #mfcc = speechpy.mfcc(signal, sampling_frequency=fs, frame_length=2.999, frame_stride=0.02,
        #     num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
        #other way of doing it:
        mfcc = speechpy.mfcc(signal, sampling_frequency=fs, frame_length=frame_length, num_filters=40, fft_length=fft_length, low_frequency=0, high_frequency=None, frame_stride=frame_length)

        
        features.append(list(mfcc[0]))
        classifications.append(instruments[i])
        
        #j = j + 1
        #print j, " / ", len(os.listdir(folders[i])) 
    print "Feature extraction:", i+1, " / ", len(folders)#, j , " / " , len(os.listdir(folders[i]))
print "time = ", time.time() - start
#featuresToText(features)

"""
for K in (1, 2, 3, 4, 5, 10, 50):
    count = 0
    for i in range(len(fluFeatures)):
        inst = knn(K, fluFeatures[i][0], features)
        #print "instrument =", inst
        if inst == 2:
            count = count + 1
        #print "Calculating: ", i, " / ", len(fluFeatures)
        
    print "With k = ", K, "accuracy = ", float(count)/float(len(fluFeatures)), count, " / ", len(fluFeatures)
"""



total1 = len(os.listdir(os.getcwd()+'\\IRMAS-TestingData\\Part1\\'))
"""
total1 = total1 + len(os.listdir(os.getcwd()+'\\IRMAS-TestingData\\Part2\\'))
total1 = total1 + len(os.listdir(os.getcwd()+'\\IRMAS-TestingData\\Part3\\'))
"""
total1 = total1 / 2

results = [0]*50

"""
model = LogisticRegression()
rfe = RFE(model, 13) #try different values
fit = rfe.fit(features, classifications)
cuttingTheFat(features, fit)
"""

#features = normalizeFeatures(features) #see whther this improves acc

j = 0
for k in (1, 2, 3, 5, 10, 20, 40, 60, 70, 75, 80, 85, 90, 95, 100, 200, 300, 400):
    correct = 0
    total = 0
    
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(features, classifications)
    
    i = 1 #for i in (1, 2, 3):
    testLoc = os.getcwd()+'\\IRMAS-TestingData\\Part' + str(i) + '\\'
    for filename in os.listdir(testLoc):
        if filename[len(filename)-4:len(filename)] == '.wav':
            #print total, " / ", total1 #todo: remove    ~~~~~~~~~~~~~~~~~~~~~~~~~~``
            textname = filename[0:len(filename)-4] + '.txt'
            textfile = open(testLoc +textname,'r')
            classification = textfile.read()
            wavename = testLoc + filename
            
            (fs, signal) = wav.read(wavename)
            signal = signal[:,0]           

            fft_length = bestFFTlength(len(signal))
            frame_length = fft_length/fs
            feature = speechpy.mfcc(signal, sampling_frequency=fs, frame_length=frame_length, num_filters=40, fft_length=fft_length, low_frequency=0, high_frequency=None, frame_stride=frame_length)
            feature = arrayAverage(feature)

            #feature = normalizeFeatures(feature)
            """
            feature = cuttingTheFat2(feature, fit)
            feature = np.array(feature)
            """
            feature = feature.reshape(1, -1) #???
            prediction = neigh.predict(feature)
            #print "prediction:", instruments[prediction], " actual: ", classification

            if prediction[0] == classification[0:3] or prediction[0] == classification[5:8]:
                correct = correct + 1
            total = total + 1
    #end for loop

    results[j] = (k, float(correct)/float(total))
    j = j+1
    print "For k = ", k, ": Total accuracy: ", correct, " / ", total, " = ", float(correct)/float(total)

print results


"""
Normal mfcc:
For k =  0 : Total accuracy:  26  /  795  =  0.0327044025157
For k =  1 : Total accuracy:  189  /  795  =  0.237735849057
For k =  2 : Total accuracy:  170  /  795  =  0.213836477987
For k =  3 : Total accuracy:  164  /  795  =  0.206289308176
For k =  4 : Total accuracy:  154  /  795  =  0.193710691824
For k =  5 : Total accuracy:  184  /  795  =  0.231446540881
For k =  6 : Total accuracy:  198  /  795  =  0.249056603774
For k =  7 : Total accuracy:  203  /  795  =  0.25534591195
For k =  8 : Total accuracy:  201  /  795  =  0.252830188679
For k =  9 : Total accuracy:  204  /  795  =  0.256603773585
For k =  10 : Total accuracy:  211  /  795  =  0.265408805031
For k =  11 : Total accuracy:  224  /  795  =  0.281761006289
For k =  12 : Total accuracy:  223  /  795  =  0.280503144654
For k =  13 : Total accuracy:  238  /  795  =  0.299371069182
For k =  14 : Total accuracy:  234  /  795  =  0.294339622642
For k =  15 : Total accuracy:  229  /  795  =  0.288050314465
For k =  16 : Total accuracy:  235  /  795  =  0.295597484277
For k =  17 : Total accuracy:  242  /  795  =  0.304402515723
For k =  18 : Total accuracy:  237  /  795  =  0.298113207547
For k =  19 : Total accuracy:  239  /  795  =  0.300628930818
For k =  20 : Total accuracy:  248  /  795  =  0.311949685535
For k =  21 : Total accuracy:  255  /  795  =  0.320754716981
For k =  22 : Total accuracy:  253  /  795  =  0.318238993711
For k =  23 : Total accuracy:  256  /  795  =  0.322012578616
For k =  24 : Total accuracy:  263  /  795  =  0.330817610063
For k =  25 : Total accuracy:  262  /  795  =  0.329559748428
For k =  26 : Total accuracy:  260  /  795  =  0.327044025157
For k =  27 : Total accuracy:  266  /  795  =  0.334591194969
For k =  28 : Total accuracy:  262  /  795  =  0.329559748428
For k =  29 : Total accuracy:  256  /  795  =  0.322012578616
For k =  30 : Total accuracy:  262  /  795  =  0.329559748428
For k =  31 : Total accuracy:  262  /  795  =  0.329559748428
For k =  32 : Total accuracy:  262  /  795  =  0.329559748428
For k =  33 : Total accuracy:  253  /  795  =  0.318238993711
For k =  34 : Total accuracy:  252  /  795  =  0.316981132075
For k =  35 : Total accuracy:  254  /  795  =  0.319496855346
For k =  36 : Total accuracy:  253  /  795  =  0.318238993711
For k =  37 : Total accuracy:  252  /  795  =  0.316981132075
For k =  38 : Total accuracy:  258  /  795  =  0.324528301887
For k =  39 : Total accuracy:  263  /  795  =  0.330817610063
For k =  40 : Total accuracy:  267  /  795  =  0.335849056604
For k =  41 : Total accuracy:  261  /  795  =  0.328301886792
For k =  42 : Total accuracy:  260  /  795  =  0.327044025157
For k =  43 : Total accuracy:  253  /  795  =  0.318238993711
For k =  44 : Total accuracy:  253  /  795  =  0.318238993711
For k =  45 : Total accuracy:  251  /  795  =  0.31572327044
For k =  46 : Total accuracy:  249  /  795  =  0.31320754717
For k =  50 : Total accuracy:  250  /  795  =  0.314465408805
"""

"""
normal ftt, normalized values, 5 features
For k =  1 : Total accuracy:  241  /  795  =  0.303144654088
For k =  2 : Total accuracy:  190  /  795  =  0.238993710692
For k =  3 : Total accuracy:  162  /  795  =  0.203773584906
For k =  5 : Total accuracy:  218  /  795  =  0.274213836478
For k =  10 : Total accuracy:  240  /  795  =  0.301886792453
For k =  20 : Total accuracy:  256  /  795  =  0.322012578616
For k =  40 : Total accuracy:  250  /  795  =  0.314465408805
For k =  60 : Total accuracy:  252  /  795  =  0.316981132075
For k =  80 : Total accuracy:  232  /  795  =  0.291823899371
For k =  100 : Total accuracy:  229  /  795  =  0.288050314465
For k =  200 : Total accuracy:  197  /  795  =  0.247798742138
"""

"""
groups' features, regular values, 13 features
For k =  1 : Total accuracy:  261  /  795  =  0.328301886792
For k =  2 : Total accuracy:  244  /  795  =  0.306918238994
For k =  3 : Total accuracy:  277  /  795  =  0.348427672956
For k =  5 : Total accuracy:  284  /  795  =  0.357232704403
For k =  10 : Total accuracy:  313  /  795  =  0.393710691824
For k =  20 : Total accuracy:  345  /  795  =  0.433962264151
For k =  40 : Total accuracy:  345  /  795  =  0.433962264151
For k =  60 : Total accuracy:  357  /  795  =  0.449056603774
For k =  80 : Total accuracy:  358  /  795  =  0.450314465409
For k =  100 : Total accuracy:  357  /  795  =  0.449056603774
For k =  200 : Total accuracy:  347  /  795  =  0.436477987421
"""

"""
group's features normalized values, 13 features
For k =  1 : Total accuracy:  232  /  795  =  0.291823899371
For k =  2 : Total accuracy:  225  /  795  =  0.283018867925
For k =  3 : Total accuracy:  229  /  795  =  0.288050314465
For k =  5 : Total accuracy:  268  /  795  =  0.337106918239
For k =  10 : Total accuracy:  303  /  795  =  0.381132075472
For k =  20 : Total accuracy:  334  /  795  =  0.420125786164
For k =  40 : Total accuracy:  331  /  795  =  0.416352201258
For k =  60 : Total accuracy:  347  /  795  =  0.436477987421
For k =  80 : Total accuracy:  326  /  795  =  0.410062893082
For k =  100 : Total accuracy:  323  /  795  =  0.406289308176
For k =  200 : Total accuracy:  310  /  795  =  0.389937106918
"""

"""
groups fs, normalized values, 12 features
For k =  1 : Total accuracy:  209  /  795  =  0.262893081761
For k =  2 : Total accuracy:  224  /  795  =  0.281761006289
For k =  3 : Total accuracy:  214  /  795  =  0.269182389937
For k =  5 : Total accuracy:  238  /  795  =  0.299371069182
For k =  10 : Total accuracy:  247  /  795  =  0.310691823899
For k =  20 : Total accuracy:  265  /  795  =  0.333333333333
For k =  40 : Total accuracy:  273  /  795  =  0.343396226415
For k =  60 : Total accuracy:  267  /  795  =  0.335849056604
For k =  80 : Total accuracy:  263  /  795  =  0.330817610063
For k =  100 : Total accuracy:  255  /  795  =  0.320754716981
For k =  200 : Total accuracy:  242  /  795  =  0.304402515723
"""

"""
groups fft, regular features, 12 feats
For k =  1 : Total accuracy:  224  /  795  =  0.281761006289
For k =  2 : Total accuracy:  240  /  795  =  0.301886792453
For k =  3 : Total accuracy:  235  /  795  =  0.295597484277
For k =  5 : Total accuracy:  250  /  795  =  0.314465408805
For k =  10 : Total accuracy:  273  /  795  =  0.343396226415
For k =  20 : Total accuracy:  301  /  795  =  0.378616352201
For k =  40 : Total accuracy:  319  /  795  =  0.401257861635
For k =  60 : Total accuracy:  317  /  795  =  0.398742138365
For k =  80 : Total accuracy:  322  /  795  =  0.405031446541
For k =  100 : Total accuracy:  320  /  795  =  0.40251572327
For k =  200 : Total accuracy:  299  /  795  =  0.376100628931
"""

"""
group, reg, 13
For k =  60 : Total accuracy:  357  /  795  =  0.449056603774
For k =  61 : Total accuracy:  353  /  795  =  0.444025157233
For k =  62 : Total accuracy:  357  /  795  =  0.449056603774
For k =  63 : Total accuracy:  358  /  795  =  0.450314465409
For k =  64 : Total accuracy:  356  /  795  =  0.447798742138
For k =  65 : Total accuracy:  358  /  795  =  0.450314465409
For k =  66 : Total accuracy:  359  /  795  =  0.451572327044
For k =  67 : Total accuracy:  358  /  795  =  0.450314465409
For k =  68 : Total accuracy:  359  /  795  =  0.451572327044
For k =  69 : Total accuracy:  356  /  795  =  0.447798742138
For k =  70 : Total accuracy:  355  /  795  =  0.446540880503
"""
"""
euclidean
For k =  70 : Total accuracy:  355  /  795  =  0.446540880503
For k =  80 : Total accuracy:  358  /  795  =  0.450314465409
For k =  90 : Total accuracy:  361  /  795  =  0.454088050314
"""

"""
manhattan
For k =  70 : Total accuracy:  359  /  795  =  0.451572327044
For k =  80 : Total accuracy:  356  /  795  =  0.447798742138
For k =  90 : Total accuracy:  362  /  795  =  0.45534591195
"""
