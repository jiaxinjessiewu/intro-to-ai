# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        evalScore = successorGameState.getScore()
        
        # calculate the total scareTimer
        sumScareTimer = 0
        for time in newScaredTimes:
            sumScareTimer += time
    
        # find the minmum distance between ghosts and newPos
        minGhostDis = 500
        for ghost in newGhostStates:
            gdis = manhattanDistance(newPos, ghost.getPosition())
            if gdis < minGhostDis:
                minGhostDis = gdis
        
        # find the minmum distance between fooods and newPos
        minFoodDis = 500
        for food in newFood.asList():
            fdis = manhattanDistance(newPos,food)
            if fdis < minFoodDis:
                minFoodDis = fdis

        # if the is no food in newPos, foodScore will have smaller weight
        foodScore = 500
        if minFoodDis > 0:
            foodScore = 1/minFoodDis

        # calculate ghostScore
        # if the minmum distance between ghost and newPos is too close(<2)
        # evalScore will be smaller
        ghostScore = minGhostDis*foodScore
        if minGhostDis < 2 and sumScareTimer == 0:
            ghostScore -= 500
    
        evalScore += foodScore + ghostScore/2.2
        
        return evalScore


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        return self.Minimax(gameState)[1]
    
    def Minimax(self, gameState):
        minMaxAction = [-99999,None]
        # pacman index = 0
        for action in gameState.getLegalActions(0):
            # find the greatest utility of minValue() and pick that maximizing action
            # starts at depth 0(max) and with 1 ghost (agentIndex=1)
            utility = self.minValue(gameState.generateSuccessor(0,action),0,1)
            if utility > minMaxAction[0]:
                minMaxAction = [utility,action]
        return minMaxAction
    
    def minValue(self,gameState,mindepth,gindex):
        # ghost index >= 1
        minLegalActions = gameState.getLegalActions(gindex)
        if gameState.isWin() or gameState.isLose() or mindepth == self.depth or len(minLegalActions) == 0:
            return self.evaluationFunction(gameState)
        minVal = 99999
        numGhost = gameState.getNumAgents() - 1
        for action in minLegalActions:
            # all ghosts are found, find the smallest utility of next depth(max)
            if gindex == numGhost:
                u = self.maxValue(gameState.generateSuccessor(gindex,action),mindepth+1,0)
                minVal = min(minVal,u)
            # continue to find smallest in current depth of other ghosts
            else:
                u = self.minValue(gameState.generateSuccessor(gindex,action),mindepth,gindex+1)
                minVal = min(minVal, u)
        return minVal
    
    def maxValue(self,gameState,maxdepth,pindex):
        maxLegalActions = gameState.getLegalActions(pindex)
        if gameState.isWin() or gameState.isLose() or maxdepth == self.depth or len(maxLegalActions) == 0:
            return self.evaluationFunction(gameState)
        maxVal = -99999
        for action in maxLegalActions:
            # find the greatest utility in minValue
            u = self.minValue(gameState.generateSuccessor(pindex,action),maxdepth,1)
            maxVal = max(maxVal, u)
        return maxVal

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        # Collect legal moves and successor states
        alpha = -float("inf")
        beta = float("inf")

        # Choose one of the best actions
        return self.prune(gameState, alpha, beta, 0, self.index)[0]


    def prune(self, gameState, alpha, beta, depth, agentIndex):
        """
        Return (best move, score) for the agent given by agentIndex in the current
        game state 
        """
        best_move = Directions.STOP
        numAgents = gameState.getNumAgents()
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return best_move, self.evaluationFunction(gameState)
            
        if agentIndex == 0: # Pacman's turn!
            v = float("-inf")
            legalMoves = gameState.getLegalActions(agentIndex)
            for move in legalMoves:
                nxt_state = gameState.generateSuccessor(agentIndex, move)
                new_index = (agentIndex + 1) % numAgents # cycle through players
                v = max(v, self.prune(nxt_state, alpha, beta, depth, new_index)[1])
                if alpha < v: 
                    alpha, best_move = v, move
                if v >= beta:
                    return best_move, v
            return best_move, v
        else: # a Ghost's turn!
            v = float("inf")
            legalMoves = gameState.getLegalActions(agentIndex)
            for move in legalMoves:
                nxt_state = gameState.generateSuccessor(agentIndex, move)
                new_index = (agentIndex + 1) % numAgents # cycle through players
                if new_index == 0:
                    v= min(v, self.prune(nxt_state, alpha, beta, depth+1, new_index)[1])
                else:
                    v = min(v, self.prune(nxt_state, alpha, beta, depth, new_index)[1])
                if beta > v: 
                    beta,best_move = v, move
                if v <= alpha: 
                    return best_move, v
            return best_move, v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.ExpectiMax(gameState)[1]
    
    def ExpectiMax(self, gameState):
        expMaxAction = [-99999,None]
        # pacman index = 0
        for action in gameState.getLegalActions(0):
            # find the greatest utility of expValue() and pick that maximizing action
            # starts at depth 0(max) and with 1 ghost (agentIndex=1)
            utility = self.expValue(gameState.generateSuccessor(0,action),0,1)
            if utility > expMaxAction[0]:
                expMaxAction[0] = utility
                expMaxAction[1] = action
        return expMaxAction
    
    def expValue(self, gameState, depths, index):
        legalActions = gameState.getLegalActions(index)
        if gameState.isWin() or gameState.isLose() or depths == self.depth or len(legalActions) == 0:
            return self.evaluationFunction(gameState)
        expVal = 0
        numGhost = gameState.getNumAgents() - 1
        if index == numGhost:
            newIndex = 0
        else:
            newIndex = index+1
        # chooses amongst the getLegalActions uniformly at random
        # calculate the probability
        prob = 1.0/len(legalActions)
        for action in legalActions:
            # all ghost are found, find the smallest utility of next depth
            if newIndex == 0:
                # calculate the expected value
                expVal += prob*self.maxValue(gameState.generateSuccessor(index,action),depths+1,newIndex)
            # continue to find smallest in current depth of other ghosts
            else:
                expVal += prob*self.expValue(gameState.generateSuccessor(index,action),depths,newIndex)
        return expVal

    def maxValue(self,gameState,maxdepth,pindex):
        maxLegalActions = gameState.getLegalActions(pindex)
        if gameState.isWin() or gameState.isLose() or maxdepth == self.depth or len(maxLegalActions) == 0:
            return self.evaluationFunction(gameState)
        maxVal = -99999
        for action in maxLegalActions:
            # find the greatest utility in expValue
            u = self.expValue(gameState.generateSuccessor(pindex,action),maxdepth,1)
            maxVal = max(maxVal, u)
        return maxVal

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      Relevant data:
        Remaining super-pellets: Pacman tends to ignore capsules unless you make this penalty really high
        Remaining food: Penalize for this to get Pacman to try to eat more
        Minimum food distance: Penalize Pacman for being far from food, but only with weight=1, as he is
            already being punished for remaining food in the first place
        Scared ghosts: Pacman gets lots of points when he eats a scared ghost
        Scary ghosts: inexplicably, Pacman does much better when he chases scary Ghosts around, too
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin(): return 99999
    if currentGameState.isLose(): return -99999
    
    # Initialize distance trackers...
    minScaredGhostDist = 0
    minScaryGhostDist = 1
    minFoodDist = 99999

    # Get game-state stats
    evalScore = currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()

    numFood = currentGameState.getNumFood()
    foodDists = [util.manhattanDistance(pos, food) for food in foods]
    foodDists = filter(lambda dist: dist > 0, foodDists)
    minFoodDist = min(foodDists) if foodDists else -99999

    caps = currentGameState.getCapsules()
    if caps:
        numCaps = len(caps)
    else:
        numCaps = 0

    # Treat scared ghosts and non-scared ghosts differently
    ghostStates = currentGameState.getGhostStates()

    # Min distance to any scared ghost
    scaredGhosts = filter(lambda ghostState: ghostState.scaredTimer > 0, ghostStates)
    if scaredGhosts:
        minScaredGhostDist = min([util.manhattanDistance(pos, ghost.getPosition()) for ghost in scaredGhosts]) 

    # Min distance to any ghost that is not scared
    scaryGhosts = filter(lambda ghostState: not bool(ghostState.scaredTimer), ghostStates)
    if scaryGhosts:
        minScaryGhostDist = min([util.manhattanDistance(pos, ghost.getPosition()) for ghost in scaryGhosts])

    if scaredGhosts:
        numFoodPenalty = -5*numFood
        numCapsPenalty = -20*numCaps
        scaredGhostPenalty = -2*minScaredGhostDist
    else:
        numFoodPenalty = -3*numFood
        numCapsPenalty = -30*numCaps
        scaredGhostPenalty = 0
        
    scaryGhostPenalty = -2*minScaryGhostDist

    return evalScore + numFoodPenalty + numCapsPenalty + scaryGhostPenalty - minFoodDist + scaredGhostPenalty

    
# Abbreviation
better = betterEvaluationFunction

