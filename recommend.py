"""Student: Hsuan-Chih, Chen"""
"""ID: W1116621"""

import numpy as np
from numpy import *

#Storing training set, iuf list, movie average list globally for looking up table
trainingSet = [[0] * 1000] * 200
iuf_list = []
movieAve_list = []

#for kNeighbor apporch
class KNeighbor:
    def __init__(self, user_id, similarity):
        self.user_id = user_id
        self.similarity = similarity

    def get_similarity(self):
        return self.similarity

    def get_user_id(self):
        return self.user_id
def calVectorlength(v):
    return np.sqrt(np.sum(np.square(v)))


def findComTerm(aList, bList):
    aNew = []
    bNew = []
    for movie in (aList):
        tempy = bList[movie]
        tempx = aList[movie]
        if tempx > 0 and tempy > 0:
            aNew.append(tempx)
            bNew.append(tempy)
    return np.array(aNew), np.array(bNew)


def calCosinePure(a, b):
    aNew, bNew = findComTerm(a, b)
    sim = np.dot(aNew, bNew)
    aDis = calVectorlength(aNew)
    bDis = calVectorlength(bNew)

    #if just one cpmmon term predict 1 or 0, 0 is better
    if aDis != 0 and bDis != 0: #and len(a_new) != 1:
        sim /= (aDis * bDis)
    return sim


def pearsonCorrelation(a, b):
    aNew, bNew = findComTerm(a, b)
    if(len(aNew) != 0):
        aAve = aNew.mean()
        bAve = bNew.mean()
        tempA = np.subtract(aNew, aAve)
        tempB = np.subtract(bNew, bAve)

        abDot = np.dot(tempA , tempB)
        aDis = calVectorlength(tempA)
        bDis = calVectorlength(tempB)
        abDis = aDis * bDis
        if abDis == 0:
            return 0
        else:
            sim = abDot/abDis
            return sim
    else:
        return 0



def loadTrainingSet():
    training = open('train.txt', 'r')
    training = training.read().strip().split('\n')
    for i, line in enumerate(training):
        trainingSet[i] = [int(x) for x in line.split()]



def cosinePure(user, movieIds):

    neighborList =[]
    count = 0
    for u in trainingSet:
        sim = calCosinePure(user, u)
        oneNeighbor = KNeighbor(count, sim)
        neighborList.append(oneNeighbor)
        count = count + 1
    #set consider k neighbor
    kNeighbor = 100
    neighborList.sort(key=lambda x: x.similarity, reverse=True)
    kNeighborList = []
    #build the list of k neighbors
    for i in range(0, len(neighborList)):
        if i < kNeighbor:
            kNeighborList.append(neighborList[i])

    ratings = []
    for movieId in movieIds:
        totalWeight = 0
        rating = 0

        for one in kNeighborList:
            ##adding IUF  for cosine
            ##adding case modify
            p = 1.5
            userId = one.get_user_id()
            userRating = trainingSet[userId][movieId]
            if userRating != 0:
                tempWeight = one.get_similarity()
                tempWeight = tempWeight*tempWeight*(p)*iuf_list[movieId]
                totalWeight += tempWeight
                rating += (tempWeight*userRating)

        if totalWeight != 0:
            rating /= totalWeight
        else:
            #give ave of the user
            total = 0
            for u in user:
                total = total + user[u]
            rating = total/len(user)
        ratings.append(rating)

    return ratingsToInt(ratings)



def pearsonPure(user, movieIds):
    """IUF"""
    #transTranSet = trainingSet
    #transTranSet = [list(x) for x in zip(*transTranSet)]
    ##IUFtrnasTranset = [[rr*iuf_list[] for rr in m if rr>0 ] for m in transTranSet]
    #row = 0
    #for m in transTranSet:
        #for rr in m:
            #if(rr >0):
                #rr *= iuf_list[row]
        #row += 1
    ##trnasTranset = [list(x) for x in zip(*transTranSet)]

    weights = [pearsonCorrelation(user, u) for
               u in trainingSet]

    """for Testing Case Modification"""
    p=1.5
    weights = [w * (np.abs(w) ** (p - 1)) for w in weights]
    userAves = [np.average([r for r in u if r > 0]) for u in trainingSet]#trainingSet#]

    #commonAverages = [np.average([comRate for i, comRate in enumerate(u) if(comRate>0 and i in user)]) for u in trainingSet]

    ratings = []
    total = 0
    for u in user:
        total = total + user[u]
    preUserAvg = total / len(user)

    for movieId in movieIds:
        totalWeight = 0
        rating = 0
        countRow = 0
        for w, u_other, userAvg in zip(weights, trainingSet, userAves):#commonAverages):#user_averages):
            u_rating = u_other[movieId]
            #for Nan data
            #if(isnan(userAvg)):
               # user_avg = 0
            ##iuf
            iuf = iuf_list[countRow]
            w *= iuf

            if u_rating != 0 :
                totalWeight += np.abs(w)
                rating += w * (u_rating - userAvg)
            ##countRow += 1
        if totalWeight != 0:
            rating = preUserAvg + (rating/totalWeight)
        else:
            #give user's his own average if there is no matched other user
            rating = preUserAvg
            #rating = 3
            #rating = movieAve_list[movie_id]
        ratings.append(rating)

    return ratingsToInt(ratings)



def ratingsToInt(ratings):
    for rating in ratings:
        rating = int(np.rint(rating))
        if rating > 5:
            rating = 5
        if rating < 1:
            rating = 1
    return ratings



def createMovieAve_list():
    transTranSet = trainingSet
    transTranSet = [list(x) for x in zip(*transTranSet)]
    for u in transTranSet:
        count = 0
        sum = 0
        ave=0
        for r in u:
            if (r > 0):
                count += 1
                sum += r
        if (count != 0):
            ave = sum / count
        else:
            ave = 0
        movieAve_list.append(ave)





def createIUFlist():
    transTranSet = trainingSet
    transTranSet = [list(x) for x in zip(*transTranSet)]

    for u in transTranSet:
        count  = 0
        for r in u:
            if(r > 0):
                count = count+1
        if(count == 0):
            iuf = 1
        else:
            iuf = np.log(200/count)
        #if(count !=1):
            #print(iuf)
        iuf_list.append(iuf)



def calCosineAdj(a, b):
    abDot = np.dot(a, b)
    aDis = calVectorlength(a)
    bDis = calVectorlength(b)
    abDis = aDis * bDis

    if(abDis==0):
        return 0
    else:
        return abDot/abDis



def cosineAdj(preM, curM,userAve):

    ##preAve = movieAve_list[preID]
    ##curAve = movieAve_list[curID]
    aNew, bNew = findComTerm(preM, curM)
    tempA = np.subtract(aNew, userAve)
    tempB = np.subtract(bNew, userAve)
    ##tempA = np.subtract(preM, preAve)
    ##tempB = np.subtract(curM, curAve)
    ##tempA, tempB = findComTerm(preM, curM)

    return calCosineAdj(tempA, tempB)
    ##return calCosineAdj(tempA, tempB)




def itemBase(user, preMovieIds):
    transTranSet = trainingSet
    transTranSet = [list(x) for x in zip(*transTranSet)]
    userMoives = list(user.keys())
    total = 0
    for u in user:
        total = total + user[u]
    preUserAvg = total / len(user)
    """"
    neighborList = []
    count = 0
    for u in transTranSet:
        sim = cals(user, u)
        oneNeighbor = KNeighbor(count, sim)
        neighborList.append(oneNeighbor)
        count = count + 1
    # set consider k neighbor
    kNeighbor = 100
    neighborList.sort(key=lambda x: x.similarity, reverse=True)
    kNeighborList = []
    # build the list of k neighbors
    for i in range(0, len(neighborList)):
        if i < kNeighbor:
            kNeighborList.append(neighborList[i])
    """


    ratings = []



    for movieId in preMovieIds:
        preMovieM = transTranSet[movieId]

        weights = [cosineAdj(preMovieM,transTranSet[i], preUserAvg)
                   for i in userMoives]
        preMovieAve = movieAve_list[movieId]
        totalWeight = 0
        rating = 0
        for w, i in zip(weights, userMoives):
            curRating = user[i]
            # Choose average movieRate as Pearson predict way
            curMoviesAve = movieAve_list[i]
            totalWeight += np.abs(w)
            rating += (w * (curRating-curMoviesAve))
        if totalWeight != 0:
            #rating /= sum_w
            rating += preMovieAve + (rating/totalWeight)
        else:
            rating = preMovieAve
            """
            total = 0
            for u in user:
                total = total + user[u]
            userAve = total / len(user)
            rating = userAve
            """
        ratings.append(rating)
    return ratingsToInt(ratings)



def predictOneUser(preUuser, preUserId, preMovieIds, results):
    if len(preMovieIds) != 0:
        ratings = itemBase(preUuser, preMovieIds)
        #ratings1 = cosinePure(preUuser, preMovieIds)
        #ratings2 = pearsonPure(preUuser, preMovieIds)
        #temp = [x*0.3 + y*0.3 + z*0.4 for x, y, z in zip(ratings1, ratings2, ratings3)]
        #ratings = ratingsToInt(temp)
        for preMovieId, rating in zip(preMovieIds, ratings):
            results.append((preUserId+1, preMovieId+1, rating))



def predictAll(testFile):
    testingData = open(testFile, 'r').read().strip().split('\n')
    testingData = [data.split() for data in testingData]
    testingData = [[int(rating) for rating in data] for data in testingData]
    curUserId = testingData[0][0] - 1
    movRatingList = {}
    movieIds = []
    results = []
    #each id need to minus 1 movies 0~999 user 0-200
    for testUserId, testMovieId, rating in testingData:
        testUserId -= 1
        testMovieId -= 1
        if  testUserId != curUserId:
            predictOneUser(
                movRatingList,
                curUserId,
                movieIds,
                results
            )
            curUserId =  testUserId
            movRatingList = {}
            movieIds = []

        if rating == 0:
            movieIds.append( testMovieId)
        else:
            movRatingList[ testMovieId] = rating
    predictOneUser(
        movRatingList,
        curUserId,
        movieIds,
        results
    )
    return results



def OutputAll(results, afile):
    fout = open(afile, 'w')
    for result in results:
        fout.write(' '.join(str(x) for x in result) + '\n')


def main():

    loadTrainingSet()
    createMovieAve_list()
    createIUFlist()
    print('test5')
    results = predictAll('test5.txt')
    OutputAll(results, 'item5.txt')
    print('test10')
    results = predictAll('test10.txt')
    OutputAll(results, 'item10.txt')
    print('test20')
    results = predictAll('test20.txt')
    OutputAll(results, 'item20.txt')


main()


