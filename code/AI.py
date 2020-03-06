#import keras

class AI():

    # normaliser les variables récupérées 
    '''
    brickPosition
    brickColor
    ballPosition
    ballDirection
    ballSpeed
    ballSize
    barPosition
    barSize
    barSpeed
    '''

    MOVE_LEFT = -1;
    MOVE_RIGHT = 1;
    DONT_MOVE = 0;

    def __init__(self):
        print("Ready")
        model = Sequential()
        model.add(Dense(units=200,input_dim=INPUTDATASIZE, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))       
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        self.model = model
        self.discountingFactor = 0.9 #gamma
        self.learningRate = 0.7 #alpha
        self.explorationFactor = 0 #epsilon
        self.inputTrain = []
        self.ouputTrain = []
        self.rewardTrain = []
        self.episodeNumber = 0
        self.finalReward = 0

    def computeMovement(self, ballPosX, ballPosY, ballAngle, ballSpeed, ballRadius):
        if(ballAngle < -0.2):
            return -1
        elif(ballAngle > 0.2):
            return 1
    
     def train(self, state):
        proba = self.model.predict(state)

        if proba < 0.4:
            action = MOVE_LEFT
        elif proba > 0.6:
            action = MOVE_RIGHT
        else:
            action = DONT_MOVE

        self.inputTrain.append(state)
        self.ouputTrain.append(action)

        nextState, reward, endOfGame = #TODO: get next frame
        self.rewardTrain(reward)
        self.finalReward += reward

        if endOfGame:
            print('At the end of episode', self.episodeNumber, 'the total reward was :', self.finalReward)
            self.episodeNumber += 1

            model.fit(x=np.vstack(self.inputTrain), y=np.vstack(self.ouputTrain), verbose=1, sample_weight=discount_rewards(self.rewardTrain, self.discountingFactor))

            self.inputTrain, self.ouputTrain, self.rewardTrain = []
            state = #TODO: nouvelle partie
            train(self, state)
            self.finalReward = 0
        else:
            train(self, nextState)
        