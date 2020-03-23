import keras
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Input
from keras.optimizers import Adam
import numpy as np
import tkinter as tk
import os
from game import Game

class AI:
    def __init__(self):
        # training variables
        self.gamma = 0.99 #discountingFactor
        self.lmbda = 0.95 #smoothing parameter
        self.maxScore = 30


    def train(self):
        game.level(np.random.choice(7))
        game.ballThrown = True
        initialState, _, _ = game.nextFrame()
    
        numberOfActions = 3
        ppoSteps = 5000
        endOfTrain = False
        bestReward = - 1000
        iters = 0
        maxIters = 50

        dummyN = np.zeros((1, 1, numberOfActions))
        dummy1 = np.zeros((1, 1, 1))

        self.actorModel = self.getActorModel(len(initialState), numberOfActions)
        self.criticModel = self.getCriticModel(len(initialState), numberOfActions)


        while not endOfTrain:

            states = [] #each state encountered 
            actions = [] #each actions taken
            values = [] #from the critic model
            masks = [] #used to separate each game
            rewards = [] #each reward associating with each state
            actionsProbs = [] #probability for each action for each turn
            actionsOnehot = [] #each action taken but in one-hot form

            # collects experiences
            for i in range(ppoSteps) :

                inputState = keras.backend.expand_dims(initialState, 0)
                #print("[BALL] ", initialState[0], initialState[1], "[BAR] ", initialState[5], initialState[6])
                actionDist = self.actorModel.predict([inputState, dummyN, dummyN, dummy1, dummyN], steps=1) #returns a probability for each action
                #if i % 100 == 0 :
                    #print(i, actionDist)
                qValue = self.criticModel.predict([inputState], steps=1) #gets the evaluation of the current state
                #print("[ACTIONS_PROBA]", actionDist)
                action = np.random.choice(numberOfActions, p=actionDist[0, :]) #get a random action according to the probas
                #print("[ACTION]", action)
                actionOnehot = np.zeros(numberOfActions)
                actionOnehot[action] = 1 #one-hot representation of the action

                game.aiAction(action)
                nextState, reward, isGameFinished = game.nextFrame()
                mask = not isGameFinished

                states.append(initialState)
                actions.append(action)
                actionsOnehot.append(actionOnehot)
                values.append(qValue)
                masks.append(mask)
                rewards.append(reward)
                actionsProbs.append(actionDist)

                if isGameFinished:
                    game.level(np.random.choice(7))
                    game.ballThrown = True
                    initialState, _, _ = game.nextFrame()
                else:
                    initialState = nextState
                
            qValue = self.criticModel.predict(inputState, steps=1)
            values.append(qValue)
            returns, advantages = self.getAdvantages(values, masks, rewards)
            '''
            fitStates = np.asarray(states)
            actorLoss = self.actorModel.fit(
                fitStates,
                actions,
                verbose=False,
                shuffle=True,
                epochs=8
            )
            '''
            fitStates = np.array(states)
            fitActionsProbs = np.array(actionsProbs)
            fitAdv = np.array(advantages)
            fitRewards = np.reshape(rewards, newshape=(-1, 1, 1))
            fitValues = np.array(values[:-1])
            self.actorModel.fit(
                [fitStates, fitActionsProbs, fitAdv, fitRewards, fitValues],
                [(np.reshape(actionsOnehot, newshape=(-1, numberOfActions)))], 
                verbose=False, shuffle=True, epochs=8)
            
            '''
            self.criticModel.fit(
                fitStates,
                actions,
                verbose=False,
                shuffle=True,
                epochs=8
            )
            '''
            self.criticModel.fit(
                [np.array(states)], 
                [np.reshape(returns, newshape=(-1,3))], 
                verbose=False, shuffle=True, epochs=8)

            #print("REWARDS ", rewards)
            testRewards = [self.testReward() for _ in range(5)]
            print("[TEST REWARDS] ", testRewards)
            avgReward = np.mean(testRewards)
            print('Average test reward=' + str(avgReward))
            if avgReward >= bestReward:
                print('Best reward=' + str(avgReward))
                self.actorModel.save('model_actor_{}.hdf5'.format(avgReward))
                #self.criticModel.save('model_critic_{}.hdf5'.format(avgReward))
                bestReward = avgReward
            if bestReward > self.maxScore:
                endOfTrain = True
            iters += 1
            game.level(np.random.choice(7))
            game.ballThrown = True
        

    # predicts the next action
    def getActorModel(self, inputDims, outputDims):
        '''
        actorModel = Sequential()
        actorModel.add(Dense(units=200,input_shape=(input_dims,), activation='relu', kernel_initializer='glorot_uniform'))
        actorModel.add(Dense(units=output_dims, activation='softmax', kernel_initializer='RandomNormal'))
        actorModel.compile(optimizer=Adam(lr=1e-4), loss='sparse_categorical_crossentropy')   
        ''' 
        
        stateInput = Input(shape=(inputDims,))
        oldpolicyProbs = Input(shape=(1, outputDims,))
        advantages = Input(shape=(1, 3,))
        rewards = Input(shape=(1, 1,))
        values = Input(shape=(1, 3,))

        # Classification block
        x = Dense(512, activation='relu', name='fc1')(stateInput)
        x = Dense(256, activation='relu', name='fc2')(x)
        outActions = Dense(outputDims, activation='softmax', name='predictions')(x)

        actorModel = Model(inputs=[stateInput, oldpolicyProbs, advantages, rewards, values],
                    outputs=[outActions])
        actorModel.compile(optimizer=Adam(lr=1e-4), loss=[self.ppoLoss(
            oldpolicyProbs,
            advantages,
            rewards,
            values)])
        
        return actorModel
        
    # evaluates the action
    def getCriticModel(self, inputDims, outputDims):
        '''
        criticModel = Sequential()
        criticModel.add(Dense(units=200,input_shape=(inputDims,), activation='relu', kernel_initializer='glorot_uniform'))
        criticModel.add(Dense(units=outputDims, activation='tanh', kernel_initializer='RandomNormal'))  
        criticModel.compile(optimizer=Adam(lr=1e-4), loss='sparse_categorical_crossentropy')
        '''     
        stateInput = Input(shape=(inputDims,))

        # Classification block
        x = Dense(512, activation='relu', name='fc1')(stateInput)
        x = Dense(256, activation='relu', name='fc2')(x)
        outActions = Dense(outputDims, activation='tanh')(x)

        criticModel = Model(inputs=[stateInput], outputs=[outActions])
        criticModel.compile(optimizer=Adam(lr=1e-4), loss='mse')
        
        return criticModel


    # computes reward over time (your action was correct if you win 3 turns after, for example)
    def getAdvantages(self, values, masks, rewards):
        advantages = []
        gae = 0 #Generalized Advantage Estimation (method used)
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lmbda * masks[i] * gae
            advantages.insert(0, gae + values[i])

        adv = np.array(advantages) - values[:-1]
        return advantages, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)


    # stabilizes the training process (to avoid unstoppable bad decision making) 
    def ppoLoss(self, oldpolicyProbs, advantages, rewards, values):
        # inner function to hide real computation
        def loss(yTrue, yPred):
            clippingVal = 0.2
            criticDiscount = 0.5
            entropyBeta = 0.001
            newpolicyProbs = yPred
            ratio = keras.backend.exp(keras.backend.log(newpolicyProbs + 1e-10) - keras.backend.log(oldpolicyProbs + 1e-10))
            p1 = ratio * advantages
            p2 = keras.backend.clip(ratio, min_value=1 - clippingVal, max_value=1 + clippingVal) * advantages
            actorLoss = -keras.backend.mean(keras.backend.minimum(p1, p2))
            criticLoss = keras.backend.mean(keras.backend.square(rewards - values))
            totalLoss = criticDiscount * criticLoss + actorLoss - entropyBeta * keras.backend.mean(
                -(newpolicyProbs * keras.backend.log(newpolicyProbs + 1e-10)))
            return totalLoss

        return loss

    # model evaluation
    def testReward(self):
        dummyN = np.zeros((1, 1, 3))
        dummy1 = np.zeros((1, 1, 1))
        game.level(np.random.choice(7))
        game.ballThrown = True
        state, _, _ = game.nextFrame()
        isGameFinished = False
        totalReward = 0
        limit = 0
        while not isGameFinished:
            inputState = keras.backend.expand_dims(state, 0)
            actionProbs = self.actorModel.predict([inputState, dummyN, dummyN, dummy1, dummyN], steps=1)
            action = np.argmax(actionProbs)
            game.aiAction(action)
            nextState, reward, isGameFinished = game.nextFrame()
            state = nextState
            totalReward += reward
            limit += 1
            if limit > 1024:
                break
        return totalReward


if __name__ == "__main__":
    ai = AI()
    # Starting up of the game
    root = tk.Tk()
    root.title("Brick Breaker")
    root.resizable(0,0)
    game = Game(root)
    game.ballThrown = True
    ai.train()

