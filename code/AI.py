import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import numpy as np
import tkinter as tk
import os
from game import Game

class AI:
    def __init__(self):
        print("Ready")
        #clipping_val = 0.2

        # training variables
        self.gamma = 0.99 #discountingFactor
        self.lmbda = 0.95 #smoothing parameter

        # game variables
        # Initialization of the window
        self.root = tk.Tk()
        self.root.title("Brick Breaker")
        self.root.resizable(0,0)
        self.root.bind("<Key>", eventsPress)
        self.root.bind("<KeyRelease>", eventsRelease)
        
# USELESS FOR THE MOMENT
    def computeMovement(self, ballPos, ballAngle, ballSpeed, ballRadius, barPos, barSpeed, barSize, shield, brickList, score):
        if random.randint(1,6) == 1:
            return random.randint(-1,1)
        if(ballPos[0]-barPos[0] < -0.05):
            return -1
        elif(ballPos[0]-barPos[0] > 0.05):
            return 1
        
    def newGame(self):
        game = Game(self.root)
        #self.root.mainloop()

    def train(self):
        initialState, initialScore, isGameFinished = game.nextFrame()
    
        numberOfActions = 3
        ppoSteps = 12
        endOfTrain = False
        bestReward = 0
        iters = 0
        maxIters = 50
        tensorBoard = TensorBoard(log_dir='./logs')

        states = [] #each state encountered 
        actions = [] #each actions taken
        values = [] #from the critic model
        masks = [] #used to separate each game
        rewards = [] #each reward associating with each state
        actionsProbs = [] #probability for each action for each turn
        actionsOnehot = [] #each action taken but in one-hot form

        self.actorModel, self.criticModel = self.getActorCriticModels(len(initialState), numberOfActions)

        while not endOfTrain:

            # collects experiences
            for i in range(ppoSteps) :
                inputState = keras.backend.expand_dims(initialState, 0)
                print("Ball: ", initialState[0], initialState[1], "Bar: ", initialState[5], initialState[6])
                actionDist = self.actorModel.predict([inputState], steps=1) #returns a probability for each action
                qValue = self.criticModel.predict([inputState], steps=1) #gets the evaluation of the current state
                print("[ACTIONS_PROBA]", actionDist)
                print("[VALUE]", qValue)
                action = np.random.choice(numberOfActions, p=actionDist[0, :]) #get a random action according to the probas
                print("[ACTION]", action)
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
                    self.newGame()
                    initialState, reward, isGameFinished = game.nextFrame()
                else:
                    initialState = nextState
                
            qValue = self.criticModel.predict(inputState, steps=1)
            values.append(qValue)
            returns, advantages = self.getAdvantages(values, masks, rewards)
            '''
            actorLoss = self.actorModel.fit(
                [states, actionsProbs, advantages, np.reshape(rewards, newshape=(-1, 1, 1)), values[:-1]],
                [(np.reshape(actionsOnehot, newshape=(-1, numberOfActions)))], verbose=True, shuffle=True, epochs=8,
                callbacks=[tensorBoard])
            
            criticLoss = self.criticModel.fit([states], [np.reshape(returns, newshape=(-1, 1))], shuffle=True, epochs=8,
                                        verbose=True, callbacks=[tensorBoard])
            '''
            data = [self.testReward() for _ in range(5)]
            print("[INFO] ", data)
            avgReward = np.mean(data)
            print('total test reward=' + str(avgReward))
            if avgReward > bestReward:
                print('best reward=' + str(avgReward))
                #actorModel.save('model_actor_{}_{}.hdf5'.format(iters, avg_reward))
                #criticModel.save('model_critic_{}_{}.hdf5'.format(iters, avg_reward))
                bestReward = avgReward
            if bestReward > 0.9 and iters > maxIters:
                endOfTrain = True
            iters += 1
        
    
    # Get the 2 models used for train
    def getActorCriticModels(self, input_dims, output_dims):
        # predicts the next action
        actorModel = Sequential()
        actorModel.add(Dense(units=200,input_dim=input_dims, activation='relu', kernel_initializer='glorot_uniform'))
        actorModel.add(Dense(units=output_dims, activation='sigmoid', kernel_initializer='RandomNormal'))
        actorModel.compile(optimizer=Adam(lr=1e-4), loss='mse')    
        '''
        actorModel.compile(optimizer=Adam(lr=1e-4), loss=[self.ppoLoss(
            oldpolicy_probs=oldpolicyProbs,
            advantages=advantages,
            rewards=rewards,
            values=values)])
        '''

        # evaluates the action
        criticModel = Sequential()
        criticModel.add(Dense(units=200,input_dim=input_dims, activation='relu', kernel_initializer='glorot_uniform'))
        criticModel.add(Dense(units=output_dims, activation='sigmoid', kernel_initializer='RandomNormal'))  
        criticModel.compile(optimizer=Adam(lr=1e-4), loss='mse')
        '''     
        criticModel.compile(optimizer=Adam(lr=1e-4), loss=[self.ppoLoss(
            oldpolicy_probs=oldpolicyrobs,
            advantages=advantages,
            rewards=rewards,
            values=values)])
        '''
        
        return actorModel, criticModel


    # computes reward over time (your action was correct if you win 3 turns after, for example)
    def getAdvantages(self, values, masks, rewards):
        print("MA ", len(rewards))
        advantages = []
        gae = 0 #Generalized Advantage Estimation (method used)
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lmbda * masks[i] * gae
            advantages.insert(0, gae + values[i])

        print("ICI ", len(advantages), len(values))
        adv = np.array(advantages) - values[:-1]
        return advantages, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)


    # stabilizes the training process (to avoid unstoppable bad decision making) 
    def ppoLoss(oldpolicy_probs, advantages, rewards, values):
        # inner function to hide real computation
        def loss(y_true, y_pred):
            newpolicy_probs = y_pred
            ratio = keras.backend.exp(keras.backend.log(newpolicy_probs + 1e-10) - keras.backend.log(oldpolicy_probs + 1e-10))
            p1 = ratio * advantages
            p2 = keras.backend.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
            actor_loss = -keras.backend.mean(keras.backend.minimum(p1, p2))
            critic_loss = keras.backend.mean(keras.backend.square(rewards - values))
            total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * keras.backend.mean(
                -(newpolicy_probs * keras.backend.log(newpolicy_probs + 1e-10)))
            return total_loss

        return loss

    # model evaluation
    def testReward(self):
        self.newGame()
        state, _, _ = game.nextFrame()
        done = False
        totalReward = 0
        print('testing...')
        limit = 0
        while not done:
            stateInput = keras.backend.expand_dims(state, 0)
            actionProbs = self.actorModel.predict([stateInput], steps=1)
            action = np.argmax(actionProbs)
            game.aiAction(action)
            nextState, reward, done = game.nextFrame()
            state = nextState
            totalReward += reward
            limit += 1
            if limit > 20:
                break
        return totalReward


# This function is called on key down.
def eventsPress(event):
    global game, hasEvent

    if event.keysym == "Left":
        game.keyPressed[0] = 1
    elif event.keysym == "Right":
        game.keyPressed[1] = 1
    elif event.keysym == "space" and not(game.textDisplayed):
        game.ballThrown = True

# This function is called on key up.
def eventsRelease(event):
    global game, hasEvent
    
    if event.keysym == "Left":
        game.keyPressed[0] = 0
    elif event.keysym == "Right":
        game.keyPressed[1] = 0

# MAIN
ai = AI()
# Starting up of the game
game = Game(ai.root)
game.ballThrown = True
ai.train()
# IDEE : passer par thread pour lancer le jeu ET le train, mais erreur au lancement
'''
if os.fork() > 0:
    print("[INFO] LANCEMENT DU JEU")
    #ai.root.mainloop()
else:
    print("[INFO] LANCEMENT DU TRAIN")
    ai.train()
'''