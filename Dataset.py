
import math
import numpy as np
from PIL import Image

class Dataset(object):

    nbInput = 0
    nbOutput = 0
    x = None
    y = None

    def __init__(self,nbInput, nbOutput):
        self.nbInput = nbInput
        self.nbOutput = nbOutput
        np.dtype('f')

    def add(self, x, y):
        if (self.x is None):
            self.x = np.array(x, ndmin=2)
        else:
            self.x = np.vstack((self.x, x))

        if (self.y is None):
            self.y = np.array(y, ndmin=2)
        else:
            self.y = np.vstack((self.y, y))

    def setPrintOption(self):
        lenMin = len(str(np.amax(self.x)))+1
        nbElem = math.sqrt(self.nbInput)
        np.set_printoptions(threshold=np.nan, linewidth=2+nbElem*lenMin, suppress=True)

    def printInput(self):
        self.setPrintOption()
        print self.x

    def printOutput(self):
        self.setPrintOption()
        print self.y

    def printValue(self):
        print "Input: " + str(self.nbInput)
        print "Output: " + str(self.nbOutput)

    def getInput(self):
        return (self.x)

    def getOutput(self):
        return (self.y)

    def imageToArray(self, path, convert):
        img = Image.open(path).convert(convert)
        ar = np.array(img)
        return (np.reshape(ar, np.size(ar)))

    def selectedOutputToArray(self, selected):
		ar = np.array([0]*self.nbOutput)
		if (selected >= 0 and selected < self.nbOutput):
			ar[selected] = 1
		return ar
