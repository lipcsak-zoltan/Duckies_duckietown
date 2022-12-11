import numpy as np 


class Log:
    def __init__(self):
        self.button = np.array([0,0,0,0,0])
        self.transition = 0 
    def setBut(self, arr):
        self.button = arr
    def setTrans(self, tran): #ensure that we are in the good range
        if tran>3:
            tran = 3
        if tran<0:
            tran=0
        self.transition=tran
    def getBut(self):
        return self.button
    def getTran(self):
        return self.transition 
    