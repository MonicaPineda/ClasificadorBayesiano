#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt


class BayesianClassifier:
    

    mean0=mean1=mean2=None
    priorProbability0=priorProbability1=priorProbability2=None
    confMatrix=None

    
    #covMatrix0,marginalProbability0,covMatrix1,marginalProbability1,covMatrix2,marginalProbability2,confMatrix=None
    
    def __init__(self,LOSS_PARAMETER=None,PERCENTAGE_OF_DATA_FOR_TRAINING=None):
        self._LOSS_PARAMETER=LOSS_PARAMETER
        self._PERCENTAGE_OF_DATA=PERCENTAGE_OF_DATA_FOR_TRAINING
    

    
    # dataSet Array
    # classNumber Number 
    # Method that separates data given a class
    def extractClass(self,dataSet, classNumber):
        set=[]
        for x in dataSet:
            if x[14]==classNumber: #donde esten las clases
                set.append(x)
        return set
    # dataSet Array
    # percentage number 
    # Method that separates the training data of the test data, given a percentage
    def extractTrainingSet(self,dataSet):
        trainingSet=[]
        size=len(dataSet)
        sizeofTrainingSet=int(self._PERCENTAGE_OF_DATA/100.0*size)
        for i in range(sizeofTrainingSet):
            ran=random.randint(0,len(dataSet)-1)
            trainingSet.append(dataSet.pop(ran))
        return np.array(trainingSet),np.array(dataSet)
    
    #trainingSet Array
    # Method that get covariance matrix and mean for bivariate normal distribution
    def getParameters(self,trainingSet):
        cleanTrainingSet = np.vstack((trainingSet[:,0],trainingSet[:,1],trainingSet[:,2],trainingSet[:,3],trainingSet[:,4],trainingSet[:,5],trainingSet[:,6],trainingSet[:,7],trainingSet[:,8],trainingSet[:,9],trainingSet[:,10],trainingSet[:,11],trainingSet[:,12],trainingSet[:,13]))#dataset without classes
        mean=np.mean(cleanTrainingSet, axis=1)
        covarianceMatrix=np.cov(cleanTrainingSet)
        return mean,covarianceMatrix
    
    
    #numberClass Number
    #arrayAllTraininSets
    #Method that returns the marginal probability for all class (prior probability)
    def priorProbability(self,numberClass,arrayAllTrainingSets):
        totalSize=0
        for i in arrayAllTrainingSets:
            totalSize+=len(i)
        return len(arrayAllTrainingSets[numberClass])/float(totalSize)
    
    #pointValue Array
    #mean Number
    #covMatrix Array
    #marginalProb Number
    #discriminant function 
    def discriminantFunction(self,pointValue,mean,covMatrix,priorProbability):
        x=np.array([pointValue[0],pointValue[1],pointValue[2],pointValue[3],pointValue[4],pointValue[5],pointValue[6],pointValue[7],pointValue[8],pointValue[9],pointValue[10],pointValue[11],pointValue[12],pointValue[13]]) #hay que llenar con el numero de variables
        part1 = 1 / ( ((2* np.pi)**(len(mean)/2)) * (np.linalg.det(covMatrix)**(1/2)) )
        part2 = (-1/2) * ((x-mean).T.dot(np.linalg.inv(covMatrix))).dot((x-mean))
        return float(part1 * np.exp(part2)*priorProbability)
    
    #traininSet Array
    #method drawing on a plot the points for each class
    def plotTrainingSet(self,trainingSet):
        if trainingSet[0][14]==0: # en donde esta el 3 va la columna donde esten las clases
            plt.plot(trainingSet[:,0],trainingSet[:,1], 'ro')
        elif trainingSet[0][14]==1:
                plt.plot(trainingSet[:,0],trainingSet[:,1], 'bo')
        else:
            plt.plot(trainingSet[:,0],trainingSet[:,1], 'go')
            
            
    def confusionMatrix(self,data,classNumber,confMatrix):
        realClass=data[14] #hay que colocar aca la columna donde este la clase 
        if realClass==classNumber:
            confMatrix[realClass][realClass]+=1
        else:
            confMatrix[realClass][classNumber]+=1
        
        
    #marginalProbVector Array
    #testSetsVector array
    #method that classifies data given a vector of marginal 
    #probabilities and array of test data and print and the reject points 
    def classify(self,testSetsVector):
        for i in testSetsVector:
            for j in i:
                likelihood0=self.discriminantFunction(j,self.mean0,self.covMatrix0,self.priorProbability0)*100000000
               # print likelihood0,"like"
                likelihood1=self.discriminantFunction(j,self.mean1,self.covMatrix1,self.priorProbability1)*100000000
                likelihood2=self.discriminantFunction(j,self.mean2,self.covMatrix2,self.priorProbability2)*100000000
                maxLikelihood=np.array([likelihood0,likelihood1,likelihood2])
                if np.amax(maxLikelihood)==likelihood0 and np.amax(maxLikelihood)>(1-self._LOSS_PARAMETER) :
                    self.confusionMatrix(j, 0, self.confMatrix)
                elif np.amax(maxLikelihood)==likelihood1 and np.amax(maxLikelihood)>(1-self._LOSS_PARAMETER):
                    self.confusionMatrix(j, 1, self.confMatrix)
                elif np.amax(maxLikelihood)==likelihood2 and np.amax(maxLikelihood)>(1-self._LOSS_PARAMETER):
                    self.confusionMatrix(j, 2, self.confMatrix)
                else:
                    self.confusionMatrix(j, 3, self.confMatrix)
    
    
    
    
    
