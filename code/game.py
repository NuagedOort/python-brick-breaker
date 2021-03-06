import tkinter as tk
import random
import math
import copy
import os
#from AI import AI

# Main class: inherit from tk.Canvas class
class Game(tk.Canvas):
    textDisplayed = False
    linesNb = 20
    seconds = 0

    # Bar properties
    barHeight = 20
    barSpeed = 10
    barWidth = 100
    barWidthEffect = barWidth + 60.0 # Not the cleanest way to normalize, but still unsure of the comportment of effect method

    # Ball property
    ballSpeed = 7
    ballRadius = 7
    ballRadiusEffect = ballRadius + 10.0

    # Shield properties
    shieldVisibility = False,

    # Bricks properties
    bricks = []
    bricksWidth = 50
    bricksHeight = 20
    bricksNbByLine = 16
    bricksColors = {
        "r": "#e74c3c",
        "g": "#2ecc71",
        "b": "#3498db",
        "t": "#1abc9c",
        "p": "#9b59b6",
        "y": "#f1c40f",
        "o": "#e67e22",
    }
    # List of brick presence per cell in the bricks grid. 0 means empty, any other float is associated to a color, hence
    # to a specific brick property
    brickListOneHot = [0] * (bricksNbByLine*linesNb)

    # Screen properties
    screenHeight = 500
    screenWidth = bricksWidth*bricksNbByLine

    # Ai properties
    #ai = AI()
    score = 0
    lastScore = 0 #to compute reward
    
    # This method initializes some attributes: the ball, the bar...
    def __init__(self, root):
        tk.Canvas.__init__(self, root, bg="#ffffff", bd=0, highlightthickness=0, relief="ridge", width=self.screenWidth, height=self.screenHeight)
        self.pack()
        self.timeContainer = self.create_text(self.screenWidth/2, self.screenHeight*4/5, text="00:00:00", fill="#cccccc", font=("Arial", 30), justify="center")
        self.shield = self.create_rectangle(0, 0, 0, 0, width=0)
        self.bar = self.create_rectangle(0, 0, 0, 0, fill="#7f8c8d", width=0)
        self.ball = self.create_oval(0, 0, 0, 0, width=0)
        self.ballNext = self.create_oval(0, 0, 0, 0, width=0, state="hidden")
        self.level(1)
        self.nextFrame()

    # This method, called each time a level is loaded or reloaded,
    # resets all the elements properties (size, position...).
    def reset(self):
        self.score = 0
        self.barWidth = 100
        self.ballRadius = 7
        self.coords(self.shield, (0, self.screenHeight-5, self.screenWidth, self.screenHeight))
        self.itemconfig(self.shield, fill=self.bricksColors["b"], state="hidden")
        self.shieldVisibility = False
        self.coords(self.bar, ((self.screenWidth - self.barWidth)/2, self.screenHeight - self.barHeight, (self.screenWidth + self.barWidth)/2, self.screenHeight))
        self.coords(self.ball, (self.screenWidth/2 - self.ballRadius, self.screenHeight - self.barHeight - 2*self.ballRadius, self.screenWidth/2 + self.ballRadius, self.screenHeight - self.barHeight))
        self.itemconfig(self.ball, fill="#2c3e50")
        self.coords(self.ballNext, tk._flatten(self.coords(self.ball)))
        self.effects = {
            "ballFire": [0, 0],
            "barTall": [0, 0],
            "ballTall": [0, 0],
            "shield": [0, -1],
        }
        self.effectsPrev = copy.deepcopy(self.effects)
        self.ballThrown = False
        self.keyPressed = [False, False]
        self.losed = False
        self.won = False
        self.ballAngle = math.radians(90)
        for brick in self.bricks:
            self.delete(brick)
            del brick

    # This method displays the Nth level by reading the corresponding file (N.txt).
    def level(self, level):
        self.reset()
        self.levelNum = level
        self.bricks = []
        try:
            file = open(str(level)+".txt")
            content = list(file.read().replace("\n", ""))[:(self.bricksNbByLine*self.linesNb)]
            file.close()
            for i, el in enumerate(content):
                col = i%self.bricksNbByLine
                line = i//self.bricksNbByLine
                if el != ".":
                    self.bricks.append(self.create_rectangle(col*self.bricksWidth, line*self.bricksHeight, (col+1)*self.bricksWidth, (line+1)*self.bricksHeight, fill=self.bricksColors[el], width=2, outline="#ffffff"))
                    # Normalisation : Color index in dict / Color dict size
                    self.brickListOneHot[i] = list(self.bricksColors.keys()).index(el) / len(self.bricksColors)
        # If there is not any more level to load, the game is finished and the end of game screen is displayed (with player time).
        except IOError:
            self.displayText("GAME ENDED IN\n" + "%02d mn %02d sec %02d" % (int(self.seconds)//60, int(self.seconds)%60, (self.seconds*100)%100), hide = False)
            return
        self.displayText("LEVEL\n"+str(self.levelNum))

    # This method, called each 1/60 of seconde, computes again
    # the properties of all elements (positions, collisions, effects...).
    def nextFrameCycle(self, numberOfFrames):
        self.won = len(self.bricks) == 0
        if self.ballThrown and not(self.textDisplayed):
            self.moveBall()

        if not(self.textDisplayed):
            self.updateTime()
            
        self.updateEffects()

        #self.aiAction()

        if not(self.textDisplayed):
            if self.won:
                self.displayText("WON!", callback = lambda: self.level(self.levelNum+1))
            elif self.losed:
                self.displayText("LOST!", callback = lambda: self.level(self.levelNum))
        

        if numberOfFrames < 5:
            numberOfFrames = numberOfFrames + 1
            self.after(int(1000/60), self.nextFrameCycle(numberOfFrames))


    def nextFrame(self):
        self.nextFrameCycle(0)

        # Compute coords
        ballCoords = self.coords(self.ball)
        barCoords = self.coords(self.bar)

        return self.getState(
            ((ballCoords[0]+ballCoords[2])/(2*self.screenWidth), (ballCoords[1]+ballCoords[3])/(2*self.screenHeight)), # Normalized BallPos
            (self.ballAngle % (2*math.pi))/(2*math.pi), 
            self.ballSpeed / 10.0, 
            self.ballRadius / self.ballRadiusEffect,
            ((barCoords[0]+barCoords[2])/(2*self.screenWidth), (barCoords[1]+barCoords[3])/(2*self.screenHeight)),     # Normalized BarPos
            self.barSpeed / 10.0,
            self.barWidth / self.barWidthEffect,
            1.0 if self.shieldVisibility else 0,    # If shield activated return 1 else, 0. Alternatively, but heavier : 1.0 if self.itemcget(self.shield, "state") == "hidden" else 0
            self.brickListOneHot,
            self.score
            )
       

    def getState(self, ballPos, ballAngle, ballSpeed, ballRadius, barPos, barSpeed, barSize, shield, brickList, score):
        ballX, ballY = ballPos
        barX, barY = barPos
        currenState = [ballX, ballY, ballAngle, ballSpeed, ballRadius, barX, barY, barSpeed, barSize, shield] + brickList


        if self.won:
            currentReward = 2*self.score
        elif self.losed:
            currentReward = self.score
        else:
            currentReward = 0
        
        '''
        if self.score == self.lastScore: # no changement
            currentReward = -0.5
        elif self.score > self.lastScore: # good movement
            currentReward = 20
            self.lastScore = self.score
        else: # bad movement
            currentReward = -10
            self.lastScore = self.score
        '''
        return currenState, currentReward, (self.losed or self.won)

    
    # This method call the game AI and act according to given results
    def aiAction(self, barMovement):

        if self.keyPressed[0]:
            self.moveBar(-game.barSpeed)
        elif self.keyPressed[1]:
            self.moveBar(game.barSpeed)
        elif barMovement == 0 :
            self.moveBar(-self.barSpeed)
        elif barMovement == 1:
            self.moveBar(self.barSpeed)
        # is barMoment == 2: do nothing

    # This method, called when left or right arrows are pressed,
    # moves "x" pixels horizontally the bar, keeping it in the screen.
    # If the ball is not thrown yet, it is also moved.
    def moveBar(self, x):
        barCoords = self.coords(self.bar)
        if barCoords[0] < 10 and x < 0:
            x = -barCoords[0]
        elif barCoords[2] > self.screenWidth - 10 and x > 0:
            x = self.screenWidth - barCoords[2]
        
        self.move(self.bar, x, 0)
        if not(self.ballThrown):
            self.move(self.ball, x, 0)

    # This method, called at each frame, moves the ball.
    # It computes:
    #     - collisions between ball and bricks/bar/edge of screen
    #     - next ball position using "ballAngle" and "ballSpeed" attributes
    #     - effects to the ball and the bar during collision with special bricks
    def moveBall(self):
        self.move(self.ballNext, self.ballSpeed*math.cos(self.ballAngle), -self.ballSpeed*math.sin(self.ballAngle))
        ballNextCoords = self.coords(self.ballNext)
        
        # Collisions computation between ball and bricks
        i = 0
        while i < len(self.bricks):
            collision = self.collision(self.ball, self.bricks[i])
            collisionNext = self.collision(self.ballNext, self.bricks[i])
            if not collisionNext:
                brickColor = self.itemcget(self.bricks[i], "fill")
                # "barTall" effect (green bricks)
                if brickColor == self.bricksColors["g"]:
                    self.effects["barTall"][0] = 1
                    self.effects["barTall"][1] = 240
                # "shield" effect (blue bricks)
                elif brickColor == self.bricksColors["b"]:
                    self.effects["shield"][0] = 1
                # "ballFire" effect (purpil bricks)
                elif brickColor == self.bricksColors["p"]:
                    self.effects["ballFire"][0] += 1
                    self.effects["ballFire"][1] = 240
                # "ballTall" effect (turquoise bricks)
                elif brickColor == self.bricksColors["t"]:
                    self.effects["ballTall"][0] = 1
                    self.effects["ballTall"][1] = 240

                if not(self.effects["ballFire"][0]):
                    if collision == 1 or collision == 3:
                        self.ballAngle = math.radians(180) - self.ballAngle     # I mean, eew
                    if collision == 2 or collision == 4:
                        self.ballAngle = -self.ballAngle
                
                # If the brick is red, it becomes orange.
                if brickColor == self.bricksColors["r"]:
                    self.itemconfig(self.bricks[i], fill=self.bricksColors["o"])
                # If the brick is orange, it becomes yellow.
                elif brickColor == self.bricksColors["o"]:
                    self.itemconfig(self.bricks[i], fill=self.bricksColors["y"])
                # If the brick is yellow (or an other color except red/orange), it is destroyed.
                else:
                    self.delete(self.bricks[i])
                    self.brickListOneHot[i] = 0.0
                    del self.bricks[i]
                self.score += 1
            i += 1

        # Collisions computation between ball and edge of screen
        if ballNextCoords[0] < 0 or ballNextCoords[2] > self.screenWidth:
            self.ballAngle = math.radians(180) - self.ballAngle
        elif ballNextCoords[1] < 0:
            self.ballAngle = -self.ballAngle
        # Collision with the bar
        elif not(self.collision(self.ballNext, self.bar)):
            ballCenter = self.coords(self.ball)[0] + self.ballRadius
            barCenter = self.coords(self.bar)[0] + self.barWidth/2
            angleX = ballCenter - barCenter
            angleOrigin = (-self.ballAngle) % (math.pi*2)
            angleComputed = math.radians(-70/(self.barWidth/2)*angleX + 90)
            self.ballAngle = (1 - (abs(angleX)/(self.barWidth/2))**0.25)*angleOrigin + ((abs(angleX)/(self.barWidth/2))**0.25)*angleComputed
        elif not(self.collision(self.ballNext, self.shield)):
            if self.effects["shield"][0]:
                self.ballAngle = -self.ballAngle
                self.effects["shield"][0] = 0
            else :
                self.losed = True
        elif (self.coords(self.ball))[0] < 0:
            self.losed = True

        self.move(self.ball, self.ballSpeed*math.cos(self.ballAngle), -self.ballSpeed*math.sin(self.ballAngle))
        self.coords(self.ballNext, tk._flatten(self.coords(self.ball)))

    # This method, called at each frame, manages the remaining time
    # for each of effects and displays them (bar and ball size...).
    def updateEffects(self):
        for key in self.effects.keys():
            if self.effects[key][1] > 0:
                self.effects[key][1] -= 1
            if self.effects[key][1] == 0:
                self.effects[key][0] = 0
        
        # "ballFire" effect allows the ball to destroy bricks without boucing on them.
        if self.effects["ballFire"][0]:
            self.itemconfig(self.ball, fill=self.bricksColors["p"])
        else:
            self.itemconfig(self.ball, fill="#2c3e50")

        # "barTall" effect increases the bar size.
        if self.effects["barTall"][0] != self.effectsPrev["barTall"][0]:
            diff = self.effects["barTall"][0] - self.effectsPrev["barTall"][0]
            self.barWidth += diff*60
            coords = self.coords(self.bar)
            self.coords(self.bar, tk._flatten((coords[0]-diff*30, coords[1], coords[2]+diff*30, coords[3])))
        # "ballTall" effect increases the ball size.
        if self.effects["ballTall"][0] != self.effectsPrev["ballTall"][0]:
            diff = self.effects["ballTall"][0] - self.effectsPrev["ballTall"][0]
            self.ballRadius += diff*10
            coords = self.coords(self.ball)
            self.coords(self.ball, tk._flatten((coords[0]-diff*10, coords[1]-diff*10, coords[2]+diff*10, coords[3]+diff*10)))
        
        # "shield" effect allows the ball to bounce once
        # at the bottom of the screen (it's like an additional life).
        if self.effects["shield"][0]:
            self.itemconfig(self.shield, fill=self.bricksColors["b"], state="normal")
            self.shieldVisibility = True
        else:
            self.itemconfig(self.shield, state="hidden")
            self.shieldVisibility = False

        self.effectsPrev = copy.deepcopy(self.effects)

    # This method updates game time (displayed in the background).
    def updateTime(self):
        self.seconds += 1/60
        self.itemconfig(self.timeContainer, text="%02d:%02d:%02d" % (int(self.seconds)//60, int(self.seconds)%60, (self.seconds*100)%100))

    # This method displays some text.
    def displayText(self, text, hide = True, callback = None):
        self.textDisplayed = True
        self.textContainer = self.create_rectangle(0, 0, self.screenWidth, self.screenHeight, fill="#ffffff", width=0, stipple="gray50")
        self.text = self.create_text(self.screenWidth/2, self.screenHeight/2, text=text, font=("Arial", 25), justify="center")
        if hide:
            self.hideText()
            #self.after(3000, self.hideText)
        if callback != None:
            self.after(3000, callback)

    # This method deletes the text display.
    def hideText(self):
        self.textDisplayed = False
        self.delete(self.textContainer)
        self.delete(self.text)

    # This method computes the relative position of 2 objects that is collisions.
    def collision(self, el1, el2):
        collisionCounter = 0

        objectCoords = self.coords(el1)
        obstacleCoords = self.coords(el2)
        
        if objectCoords[2] < obstacleCoords[0] + 5:
            collisionCounter = 1
        if objectCoords[3] < obstacleCoords[1] + 5:
            collisionCounter = 2
        if objectCoords[0] > obstacleCoords[2] - 5:
            collisionCounter = 3
        if objectCoords[1] > obstacleCoords[3] - 5:
            collisionCounter = 4
                
        return collisionCounter

