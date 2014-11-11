 #!/usr/bin/env python
# -*- coding: utf-8 -*-
from BayesianClassifier import BayesianClassifier
import numpy as np
import matplotlib.pyplot as plt
import time
from decimal import Decimal




#LOSS_PARAMETER_INIT= Decimal(0.99999999797) #rejection constant init                         
#LOSS_PARAMETER_END=  Decimal(0.999999999) #rejection constant end

#LOSS_PARAMETER_INIT= Decimal(0.999999999) #rejection constant init  
LOSS_PARAMETER_INIT=  Decimal(0.99999999999900002212172) #rejection constant init                         
LOSS_PARAMETER_END=  Decimal(1) #rejection constant end

ITERATIONS_NUMBER=100
TESTS_NUMBER=10 #number of times to classify per time
PERCENTAGE_OF_DATA_FOR_TRAINING=80 #reject percentage constant
DATAFILE='./encuesta.csv'

dataSet=None  #load data
accumulatedConfMatrix=None
lambdaVsNC0=lambdaVsNC1=lambdaVsNC2=None
lambdaVsCor0=lambdaVsCor1=lambdaVsCor2=None
lambdaVsInCor0=lambdaVsInCor1=lambdaVsInCor2=None
lambdaVsSc=None
class0=class1=class2=None


# Method that load data from csv and store them in a numpy array
def loadData():
    data = np.genfromtxt(DATAFILE, delimiter=',')
    return (data)

def classify(LOSS_PARAMETER):
    global accumulatedConfMatrix,PERCENTAGE_OF_DATA_FOR_TRAINING,dataSet
    cl=BayesianClassifier(LOSS_PARAMETER,PERCENTAGE_OF_DATA_FOR_TRAINING)#class instance
    class0=cl.extractClass(dataSet, 0)#extract data for each class
    class1=cl.extractClass(dataSet, 1)
    class2=cl.extractClass(dataSet, 2)
    [trainingSet0,testSet0]=cl.extractTrainingSet(class0)#separate data for each training 
    [trainingSet1,testSet1]=cl.extractTrainingSet(class1)
    [trainingSet2,testSet2]=cl.extractTrainingSet(class2)
    
    
    [cl.mean0,cl.covMatrix0]=cl.getParameters(trainingSet0)#the parameters of a multivariate distribution are calculated for each class (mean, covariance)
    [cl.mean1,cl.covMatrix1]=cl.getParameters(trainingSet1)
    [cl.mean2,cl.covMatrix2]=cl.getParameters(trainingSet2)
    
    cl.confMatrix=np.zeros((3,4), dtype=np.int)    #initialize confusion matrix
    
    cl.priorProbability0=cl.priorProbability(0,[trainingSet0,trainingSet1,trainingSet2]) #calculate the prior probability for each class
    cl.priorProbability1=cl.priorProbability(1,[trainingSet0,trainingSet1,trainingSet2])
    cl.priorProbability2=cl.priorProbability(2,[trainingSet0,trainingSet1,trainingSet2])
    
    testSetsVector=[testSet0,testSet1,testSet2]
    
    cl.classify(testSetsVector) #classify!
    accumulatedConfMatrix=np.add(accumulatedConfMatrix,cl.confMatrix) #acumulate results
    


def percentualConfusionMatrix():
    n=np.sum(accumulatedConfMatrix,axis=1).astype(float)
    return ((accumulatedConfMatrix.T/n)*100).T
    
   
def lambdaVsNC(percentualConfMatrix,lossParameter,classNumber):
    return np.array([lossParameter,percentualConfMatrix[classNumber,3]])

def lambdaVsCor(percentualConfMatrix,lossParameter,classNumber):
    return np.array([lossParameter,percentualConfMatrix[classNumber,classNumber]])

def lambdaVsInCor(percentualConfMatrix,lossParameter,classNumber):
    aux=100-percentualConfMatrix[classNumber,classNumber]-percentualConfMatrix[classNumber,3]
    return np.array([lossParameter,aux])

def lambdaVsScore(percentualConfMatrix,lossParameter):
    incorrect=Decimal(str(lambdaVsInCor(percentualConfMatrix,lossParameter,2)[1]))
    rejected=Decimal(str(lambdaVsNC(percentualConfMatrix,lossParameter,2)[1]))
    sum=incorrect+rejected*Decimal(0.5)
    return np.array([lossParameter,1/sum])
    

def myLineSpace(init,end,step):
    aux=end-init
    variation=aux/np.float128(step)
    array=np.array([init],dtype=np.float128)
    for i in range(step):
        init+=variation
        array=np.vstack((array,init))
    return array


def runAnalisis():
    totalTime=time.time()
    global LOSS_PARAMETER_INIT,LOSS_PARAMETER_END,ITERATIONS_NUMBER,accumulatedConfMatrix,lambdaVsNC0,lambdaVsNC1,lambdaVsNC2,lambdaVsCor0,lambdaVsCor1,lambdaVsCor2,lambdaVsInCor0,lambdaVsInCor1,lambdaVsInCor2,lambdaVsSc
    LOSS_PARAMETER=LOSS_PARAMETER_INIT
    INCREMENT=(LOSS_PARAMETER_END-LOSS_PARAMETER_INIT)/ITERATIONS_NUMBER
    print INCREMENT,"increment"
    i=0
    while LOSS_PARAMETER<=LOSS_PARAMETER_END:

        accumulatedConfMatrix=np.zeros((3,4), dtype=np.longdouble) #restart acumulated Confusion Matrix
        millis = int(round(time.time() * 1000))
        for j in range(TESTS_NUMBER):
            classify(LOSS_PARAMETER)
        print i+1, "of",ITERATIONS_NUMBER    
        print "time taken=",int(round(time.time() * 1000))-millis, "ms"
        percentualMatrix=percentualConfusionMatrix()
        
        print percentualMatrix,"\n --------------------------\n"
        
        if(lambdaVsNC0==None or lambdaVsCor0==None):
            lambdaVsNC0=lambdaVsNC(percentualMatrix,LOSS_PARAMETER,0)#matrix not classify
            lambdaVsNC1=lambdaVsNC(percentualMatrix,LOSS_PARAMETER,1)
            lambdaVsNC2=lambdaVsNC(percentualMatrix,LOSS_PARAMETER,2)
            lambdaVsCor0=lambdaVsCor(percentualMatrix,LOSS_PARAMETER,0)#matrix successfully classified
            lambdaVsCor1=lambdaVsCor(percentualMatrix,LOSS_PARAMETER,1)
            lambdaVsCor2=lambdaVsCor(percentualMatrix,LOSS_PARAMETER,2)
            lambdaVsInCor0=lambdaVsInCor(percentualMatrix,LOSS_PARAMETER,0)#matrix unsuccessfully classified
            lambdaVsInCor1=lambdaVsInCor(percentualMatrix,LOSS_PARAMETER,1)
            lambdaVsInCor2=lambdaVsInCor(percentualMatrix,LOSS_PARAMETER,2)
            lambdaVsSc=lambdaVsScore(percentualMatrix,LOSS_PARAMETER)
        else:
            lambdaVsNC0=np.vstack((lambdaVsNC0, lambdaVsNC(percentualMatrix,LOSS_PARAMETER,0)))#matrix not classify
            lambdaVsNC1=np.vstack((lambdaVsNC1, lambdaVsNC(percentualMatrix,LOSS_PARAMETER,1)))
            lambdaVsNC2=np.vstack((lambdaVsNC2, lambdaVsNC(percentualMatrix,LOSS_PARAMETER,2)))
            lambdaVsCor0=np.vstack((lambdaVsCor0, lambdaVsCor(percentualMatrix,LOSS_PARAMETER,0)))#matrix successfully classified
            lambdaVsCor1=np.vstack((lambdaVsCor1, lambdaVsCor(percentualMatrix,LOSS_PARAMETER,1)))
            lambdaVsCor2=np.vstack((lambdaVsCor2, lambdaVsCor(percentualMatrix,LOSS_PARAMETER,2)))
            lambdaVsInCor0=np.vstack((lambdaVsInCor0, lambdaVsInCor(percentualMatrix,LOSS_PARAMETER,0)))#matrix unsuccessfully classified
            lambdaVsInCor1=np.vstack((lambdaVsInCor1, lambdaVsInCor(percentualMatrix,LOSS_PARAMETER,1)))
            lambdaVsInCor2=np.vstack((lambdaVsInCor2, lambdaVsInCor(percentualMatrix,LOSS_PARAMETER,2)))
            lambdaVsSc=np.vstack((lambdaVsSc, lambdaVsScore(percentualMatrix,LOSS_PARAMETER)))
        print LOSS_PARAMETER, "loss parameter (lambda)"
        LOSS_PARAMETER+=INCREMENT
        i+=1

    diff=int(round(time.time()-totalTime))
    hours, minutes, seconds = diff/3600 ,(diff / 60)%60, diff % 60
    
    print "**********\nTotal Time Taken: "+ str(hours) + " h, "+str(minutes)+ " m, "+str(seconds)+" secs"
        

def plotAnalisis():
    plt.plot(lambdaVsNC0[:,0],lambdaVsNC0[:,1],label='Clase 0 (Bajo Riesgo)')
    plt.plot(lambdaVsNC1[:,0],lambdaVsNC1[:,1],label='Clase 1 (Riesgo Moderado)')
    plt.plot(lambdaVsNC2[:,0],lambdaVsNC2[:,1],label='Clase 2 (Alto Riesgo)')
    plt.title('$\lambda$ vs Region de rechazo')
    plt.xlabel('Parametro $\lambda$ ')
    plt.ylabel('Rechazos (%)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
    plt.grid()
    plt.show()
    
    plt.plot(lambdaVsCor0[:,0],lambdaVsCor0[:,1],label='Clase 0 (Bajo Riesgo)')
    plt.plot(lambdaVsCor1[:,0],lambdaVsCor1[:,1],label='Clase 1 (Riesgo Moderado)')
    plt.plot(lambdaVsCor2[:,0],lambdaVsCor2[:,1],label='Clase 2 (Alto Riesgo)')
    plt.title('$\lambda$ vs Aciertos')
    plt.xlabel('Parametro $\lambda$ ')
    plt.ylabel('Aciertos (%)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
    plt.grid()
    plt.show()
    
    plt.plot(lambdaVsInCor0[:,0],lambdaVsInCor0[:,1],label='Clase 0 (Bajo Riesgo)')
    plt.plot(lambdaVsInCor1[:,0],lambdaVsInCor1[:,1],label='Clase 1 (Riesgo Moderado)')
    plt.plot(lambdaVsInCor2[:,0],lambdaVsInCor2[:,1],label='Clase 2 (Alto Riesgo)')
    plt.title('$\lambda$ vs Fallos')
    plt.xlabel('Parametro $\lambda$ ')
    plt.ylabel('Fallos (%)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
    plt.grid()
    plt.show()
    

    plt.plot(lambdaVsSc[:,0],lambdaVsSc[:,1],label='Clase 2 (Alto Riesgo)')
    plt.title('$\lambda$ vs Score')
    plt.xlabel('Parametro $\lambda$ ')
    plt.ylabel('Score')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)
    plt.grid()
    plt.show()
    
    
def saveDataFiles():
    np.savetxt("lambdaVsNC0.csv", lambdaVsNC0,fmt='%0.23f', delimiter=",")
    np.savetxt("lambdaVsNC1.csv", lambdaVsNC1,fmt='%0.23f', delimiter=",")
    np.savetxt("lambdaVsNC2.csv", lambdaVsNC2,fmt='%0.23f', delimiter=",")    
    np.savetxt("lambdaVsCor0.csv", lambdaVsCor0,fmt='%0.23f', delimiter=",")
    np.savetxt("lambdaVsCor1.csv", lambdaVsCor1,fmt='%0.23f', delimiter=",")
    np.savetxt("lambdaVsCor2.csv", lambdaVsCor2,fmt='%0.23f', delimiter=",")
    np.savetxt("lambdaVsInCor0.csv", lambdaVsInCor0,fmt='%0.23f', delimiter=",")
    np.savetxt("lambdaVsInCor1.csv", lambdaVsInCor1,fmt='%0.23f', delimiter=",")
    np.savetxt("lambdaVsInCor2.csv", lambdaVsInCor2,fmt='%0.23f', delimiter=",")
    np.savetxt("lambdaVsSc.csv", lambdaVsSc,fmt='%0.23f', delimiter=",")

    
        
    
def main():
    global dataSet
    dataSet=loadData() #load the dataset
    runAnalisis()
    saveDataFiles()
    plotAnalisis()  


if __name__== "__main__":
    main() 
    



    
