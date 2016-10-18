import numpy as np
import getdat
import random

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class fullyConnected(object):
    def __init__(self):
        self.layers = []

    def creatLayer(self,numberOfInits, activationFunction = None):
        l = layer(numberOfInits,activationFunction)
        self.layers.append(l)

        return self

    def display(self):
        print len(self.layers)

    def training(self,training_set, test_set, epochs, alpha):
        for l in range(len(self.layers) - 1):
            self.layers[l].prevTraining( self.layers[l+1].getLen())

        for epoch in range(epochs):
            random.shuffle(training_set)

            for x,y in training_set:
                self.backprop(x,y,alpha)

            print "epoch {0} : {1} / {2}".format(epoch, self.percentTest(test_set),len(test_set))

    def backprop(self, x, y, alpha):
        for l in self.layers[:-1]:
            x = l.feedforward(x)

        x = sigmoid(x)

        delta = (x - y) * x * (1 - x)

        for l in xrange(2,len(self.layers)):
            delta = self.layers[-l].backprop(delta, alpha)

    def predict(self,x):
        for l in self.layers[:-1]:
            x = l.feedforward(x)

        return np.argmax(x)

    def percentTest(self,test_set):
        return sum(int(self.predict(x) == np.argmax(y)) for x,y in test_set)

class layer(object):
    """
    class layer:
        inits : the value of layer before activation function
        activation : the value of layer after activation function
        weight : the weight connect this layer to the next layer
        bias : the bias of next layer
        activationFunction (string) : "sigmoid" , "relu" or None the function use for activation function
    """
    def __init__(self,length,act = None):
        self.inits = np.random.randn(length,1)
        self.activation = np.random.randn(length,1)
        #self.weight
        #self.bias
        self.activationFunction = act;

    def getLen(self):
        return len(self.inits)

    def prevTraining(self,lengthOfNextLayer):
        self.weight = 0.01 * np.random.randn(lengthOfNextLayer, len(self.inits))
        self.bias = 0.01 * np.random.randn(lengthOfNextLayer,1)

    def feedforward(self,x):
        self.inits = x;

        if(self.activationFunction == "sigmoid"):
            x = sigmoid(x)

        if(self.activationFunction == "relu"):
            x = (x > 0) * x

        self.activation = x;
        x = np.dot(self.weight, x) + self.bias

        return x;

    def backprop(self,delta, alpha):
        if(self.activationFunction == "sigmoid"):
            sp = self.activation * ( 1 - self.activation)

        if(self.activationFunction == "relu"):
            sp = (self.inits > 0) * 1

        thisDelta = np.dot(self.weight.T, delta) * sp
        delWeight = np.dot(delta, self.activation.T)

        self.weight = self.weight - alpha * delWeight
        self.bias = self.bias - alpha * delta

        return thisDelta

f = fullyConnected()
f.creatLayer(400).creatLayer(50,"relu").creatLayer(50,"relu").creatLayer(2)
training_set,test_set = getdat.getdata()

f.training(training_set,test_set,300,0.1)
