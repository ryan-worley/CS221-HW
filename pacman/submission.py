from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


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

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)

    def recurse(gameState, agentIndex, depth, action):

        # Case where game is over, return the score as the action
        if depth == 0:
            return self.evaluationFunction(gameState), action

        # Case where you return total score value if game is over
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore(), action

        # Pull out all legal moves that pacman can take in this situation
        legalmoves = gameState.getLegalActions(agentIndex)

        # If agent is not last agent
        if agentIndex < gameState.getNumAgents() - 1:
            rewards = [recurse(gameState.generateSuccessor(agentIndex, act), agentIndex + 1, depth, action)[0] for
                       act in legalmoves]

            # Code for pacman index
            if agentIndex == 0:
                max_reward = max(rewards)
                best_indices = [index for index, value in enumerate(legalmoves) if rewards[index] == max_reward]
                best_index = random.choice(best_indices)
                return max_reward, legalmoves[best_index]

            # Code for agent index, not last
            else:
                min_reward = min(rewards)
                best_indices = [index for index, value in enumerate(legalmoves) if rewards[index] == min_reward]
                best_index = random.choice(best_indices)
                return min_reward, legalmoves[best_index]

        # Code for last index, reset the recursive loop, change depth and index
        if agentIndex == gameState.getNumAgents() - 1:
            rewards = [recurse(gameState.generateSuccessor(agentIndex, act), 0, depth - 1, action)[0] for
                       act in legalmoves]
            min_reward = min(rewards)
            best_indices = [index for index, value in enumerate(legalmoves) if rewards[index] == min_reward]
            best_index = random.choice(best_indices)
            return min_reward, legalmoves[best_index]

    reward, action = recurse(gameState, self.index, self.depth, None)
    return action
    # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """
  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 49 lines of code, but don't worry if you deviate from this)
    def recurse(gameState, agentIndex, depth, action, alpha, beta):

        # Case where game is over, return the score as the action
        if depth == 0:
            return self.evaluationFunction(gameState), action

        # Case where you return total score value if game is over
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore(), action

        legalmoves = gameState.getLegalActions(agentIndex)
        numagents = gameState.getNumAgents()

        # If the agent is the opponent
        if 1 <= agentIndex <= numagents - 2:
            possible_reward = []
            for possible_action in legalmoves:
                # Recurse
                reward = recurse(gameState.generateSuccessor(agentIndex, possible_action), agentIndex + 1, depth,
                                 possible_action, alpha, beta)[0]
                if reward > beta:
                    break
                possible_reward.append(reward)
            beta = max(possible_reward)
            min_reward = min(possible_reward)
            best_indices = [index for index, value in enumerate(legalmoves) if possible_reward[index] == min_reward]
            best_index = random.choice(best_indices)
            return min_reward, legalmoves[best_index]

        # If the agent is the last opponent, reset the agent count to pacman, increase the depth
        elif agentIndex == numagents - 1:
            possible_reward = []
            for possible_action in legalmoves:
                # Recurse, increment
                reward = recurse(gameState.generateSuccessor(agentIndex, possible_action), 0, depth
                                 - 1, possible_action, alpha, beta)[0]
                if reward > beta:
                    break
                possible_reward.append(reward)
            beta = max(possible_reward)
            min_reward = min(possible_reward)
            best_indices = [index for index, value in enumerate(legalmoves) if possible_reward[index] == min_reward]
            best_index = random.choice(best_indices)
            return min_reward, legalmoves[best_index]


        # If the agent is pacman
        elif agentIndex == 0:
            possible_reward = []
            for possible_action in legalmoves:
                # Recurse with pacman
                reward = recurse(gameState.generateSuccessor(agentIndex, possible_action), agentIndex + 1, depth,
                                 possible_action, alpha, beta)[0]
                # Break recursion tree if alpha is not met
                if reward < alpha:
                    break
                possible_reward.append(reward)
            # reset the alpha value in the search
            alpha = min(possible_reward)
            max_reward = max(possible_reward)
            best_indices = [index for index, value in enumerate(legalmoves) if possible_reward[index] == max_reward]
            best_index = random.choice(best_indices)
            return max_reward, legalmoves[best_index]

    final_reward, action = recurse(gameState, self.index, self.depth, None, float('-inf'), float('inf'))
    return action



    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
    def recurse(gameState, agentIndex, depth, action):

        # Case where game is over, return the score as the action
        if depth == 0:
            return self.evaluationFunction(gameState), action

        # Case where you return total score value if game is over
        if gameState.isWin() or gameState.isLose():
            return gameState.getScore(), action

        legalmoves = gameState.getLegalActions(agentIndex)

        numagents = gameState.getNumAgents()
        if agentIndex < numagents - 1:
            rewards = [recurse(gameState.generateSuccessor(agentIndex, act), agentIndex + 1, depth, action)[0] for
                       act in legalmoves]

            # Code for pacman index
            if agentIndex == 0:
                max_reward = max(rewards)
                best_indices = [index for index, value in enumerate(legalmoves) if rewards[index] == max_reward]
                best_index = random.choice(best_indices)
                return max_reward, legalmoves[best_index]

            # Code for agent index
            else:
                average_reward = sum(rewards)/len(rewards)
                random_action = random.choice(legalmoves)
                return average_reward, random_action

        # Code for last index, reset the recursive loop, change depth and index
        if agentIndex == numagents - 1:
            rewards = [recurse(gameState.generateSuccessor(agentIndex, act), 0, depth - 1, action)[0] for
                       act in legalmoves]
            average_reward = sum(rewards)/len(rewards)
            random_action = random.choice(legalmoves)
            return average_reward, random_action

    reward, action = recurse(gameState, self.index, self.depth, None)
    return action
    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
    """
    Better evaluate
    Features:
    GameState: What the score of the game is at one time
    Food Vicinity: How much food is in the surrounding 5 squares
    ghostdistance: How far ghosts are away from player, manhattan distance
    pillfeature: How far pill features are from player, manhattan distance
    numfood: number of food left on the board
    numpills: number of pills left on the board
    """

    # BEGIN_YOUR_CODE (our solution is 26 lines of code, but don't worry if you deviate from this)
    # Useful information you can extract from a GameState (pacman.py)

    # Get new and old position of pacman
    position = currentGameState.getPacmanPosition()

    # Get new and old food graphs
    oldFood = currentGameState.getFood()
    numFood = currentGameState.getNumFood()

    # Count food definition, counts how much food is in adjacent squares
    def count(food, position):
        numrows = len(list(food))
        numcolumns = len(list(food[0]))
        counter = 0
        x, y = position
        xs = []
        ys = []
        for i in range(5):
            xrange = (i - 1 + x)
            yrange = (i - 1 + y)
            if 0 <= xrange <= 19:
                xs.append(xrange)
            if 0 <= yrange <= 6:
                ys.append(yrange)
        for x in xs:
            for y in ys:
                if food[x][y]:
                    counter += 1
        return counter

    # Get average distance from pacman to some object(s) in list
    def avg_distance(items, pacman):
        d = 0
        for x, y in items:
            d += util.manhattanDistance((x, y), pacman)
        return d

    # Get number of old and new food in game
    near_food_num = count(oldFood, position)
    food_vicinity = near_food_num
    if food_vicinity == 0:
        food_vicinity = -2

    # Ghost Distance
    ghost_positions = currentGameState.getGhostPositions()
    ghost_distance = 1/avg_distance(ghost_positions, position)

    # Get rid of ghost distance feature if less than 10 food remaining in the game
    if numFood < 10:
        ghost_distance = 0

    # pill distance
    capsules = currentGameState.getCapsules()
    pill_distance = avg_distance(capsules, position)

    # If there are no more pills left in the game, set pill feature to be 0
    if pill_distance == 0:
        pillfeature = 0
    else:
        pillfeature = 1/pill_distance

    # Calculate number of pills left in the game
    numpills = len(capsules)

    # Create feature and weight vector, multiply for the evaluation function
    phi = [currentGameState.getScore(), food_vicinity, ghost_distance, pillfeature, numFood, numpills]
    w = [1, 3.8, -15, 1, -8, -30]
    w = list(map(float, w))
    evaluation = 0
    for index in range(len(w)):
        evaluation += w[index]*phi[index]
    return evaluation
    # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction
