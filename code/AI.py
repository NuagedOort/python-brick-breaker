import keras
import numpy as np

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
    (barSpeed)
    '''

    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    DONT_MOVE = 0.5

    def __init__(self):
        print("Ready")
        clipping_val = 0.2
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
        self.critic_discount = 0.5
        self.entropy_beta = 0.001
        self.gamma = 0.99
        self.lmbda = 0.95

    def computeMovement(self, ballPos, ballAngle, ballSpeed, ballRadius, barPos, barSpeed, barSize, shield, brickList, score):
        if(ballPos[0]-barPos[0] < -0.2):
            return -1
        elif(ballPos[0]-barPos[0] > 0.2):
            return 1

    def getActorCriticModels(self, input_dims, output_dims):
    
        actorModel = Sequential()
        actorModel.add(Dense(units=200,input_dim=input_dims, activation='relu', kernel_initializer='glorot_uniform'))
        actorModel.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))       
        actorModel.compile(loss='mse', optimizer='adam', metrics=['mae'])

        criticModel = Sequential()
        criticModel.add(Dense(units=200,input_dim=input_dims, activation='relu', kernel_initializer='glorot_uniform'))
        criticModel.add(Dense(units=1, activation='tanh', kernel_initializer='RandomNormal'))       
        criticModel.compile(loss='mse', optimizer='adam', metrics=['mae'])
        
        return actorModel, criticModel

    
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

    
    def train(self):
        initalState = #TODO: get initial state

        numberOfActions = 3
        ppoSteps = 128
        endOfTrain = False

        states = []
        actions = []
        values = []
        masks = []
        rewards = []
        actionsProbs = []
        actionsOnehot = []

        actorModel, criticModel = getActorCriticModels(initialState.length, numberOfActions)

        while not endOfTrain:

            for i in range ppoSteps:
                inputState = keras.backend.expand_dims(initalState, 0)
                actionDist = actorModel.predict([inputState])
                qValue = criticModel.predict([inputState])

                action = np.random.choice(numberOfActions, p=actionDist[0, :])
                actionsOnehot = np.zeros(numberOfActions)
                actionsOnehot[action] = 1

                nextState, reward, isGameFinished = #TODO: get next state according to action
                mask = not isGameFinished

                states.append(initalState)
                actions.append(action)
                actionsOnehot.append(actionsOnehot)
                values.append(qValue)
                masks.append(mask)
                rewards.append(reward)
                actionsProbs.append(actionDist)

                if isGameFinished:
                    initialState = #TODO: get initial state
                else:
                    initialStete = nextState

                
            qValue = criticModel.predict(inputState)
            values.append(qValue)
            returns, advantages = getAdvantages(values, masks, rewards)
            actorLoss = actorModel.fit(
                [states, actionsProbs, advantages, np.reshape(rewards, newshape=(-1, 1, 1)), values[:-1]],
                [(np.reshape(actionsOnehot, newshape=(-1, numberOfActions)))], verbose=True, shuffle=True, epochs=8,
                callbacks=[tensor_board])
            criticLoss = criticModel.fit([states], [np.reshape(returns, newshape=(-1, 1))], shuffle=True, epochs=8,
                                        verbose=True, callbacks=[tensor_board])

            avgReward = np.mean([testReward() for _ in range(5)])
            print('total test reward=' + str(avg_reward))
            if avg_reward > best_reward:
                print('best reward=' + str(avg_reward))
                model_actor.save('model_actor_{}_{}.hdf5'.format(iters, avg_reward))
                model_critic.save('model_critic_{}_{}.hdf5'.format(iters, avg_reward))
                best_reward = avg_reward
            if best_reward > 0.9 or iters > max_iters:
                target_reached = True
            iters += 1
            env.reset()

    def getAdvantages(self, values, masks, rewards):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lmbda * masks[i] * gae
            advantages.insert(0, gae + values[i])

        adv = np.array(advantages) - values[:-1]
        return advantages, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    
    def loss(self, y_true, y_pred):
        newpolicy_probs = y_pred
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
            -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        return total_loss
        