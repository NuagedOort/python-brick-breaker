#import keras

class AI():

    def __init__(self):
        print("Ready")

    def computeMovement(self, ballPosX, ballPosY, ballAngle, ballSpeed, ballRadius):
        if(ballAngle < -0.2):
            return -1
        elif(ballAngle > 0.2):
            return 1
        