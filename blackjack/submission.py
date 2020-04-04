import util, math, random
from collections import defaultdict
from util import ValueIteration
import copy

############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    """
    Use some sort of numberline counter example to counter example. Basis of example is that
    Get more reward for smaller probability edges stemming from certain action. Having noise
    in the problem means that there is a 50% chance that a higher reward is gained, due to noise
    skewing the reward to a higher value.
    """
    # Return a value of any type capturing the start state of the MDP.
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0
        # END_YOUR_CODE

    # Return a list of strings representing actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        action = ['add num', 'subtract num']
        return action
        # END_YOUR_CODE

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # Remember that if |state| is an end state, you should return an empty list [].
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        results = []
        if state == 1 or state == -1:
            return results
        if action == 'add num':
            results.append((state + 1, .9, 80))  # Adds to 100 percent
            results.append((state - 1, .1, 180))
        if action == 'subtract num':
            results.append((state - 1, .8, 50))  # Adds to 100 percent for subtract num
            results.append((state + 1, .2, 400))

        return results
        # END_YOUR_CODE

    # Set the discount factor (float or integer) for your counterexample MDP.
    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 1
        # END_YOUR_CODE

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 53 lines of code, but don't worry if you deviate from this)

        result = []
        handValue, peek_index, numCards = state

        # If numcards tuple is set to none, end state is reached and we return and empty result
        if numCards is None:
            return []

        # Do front end calculation of probability of each card being drawn
        listNumCards = list(numCards)
        probability = [float(i) / sum(listNumCards) for i in listNumCards]

        # If the action is take, we enter here
        if action == 'Take':
            # IF the peek_index hasn't been used, we find probability of next cards being drawn
            if peek_index is None:
                for card in self.cardValues:
                    card_index = self.cardValues.index(card)
                    if probability[card_index] != 0:
                        list_newCardNums = copy.copy(listNumCards)
                        list_newCardNums[card_index] += -1
                        newHandValue = handValue + card
                        if newHandValue > self.threshold:
                            newState = (newHandValue, None, None)
                            result.append((newState, probability[card_index], 0))
                        else:
                            if sum(list_newCardNums) == 0:
                                newState = (newHandValue, None, None)
                                result.append((newState, probability[card_index], newHandValue))
                            else:
                                newState = (newHandValue, None, tuple(list_newCardNums))
                                result.append((newState, probability[card_index], 0))
                #return result

            # If peek index is used, we replace the newCard with the card the peek card
            else:
                newCard = self.cardValues[peek_index]
                list_newCardNums = copy.copy(listNumCards)
                list_newCardNums[peek_index] += -1
                newHandValue = handValue + newCard

                # Empty deck, following new hand
                if sum(list_newCardNums) == 0:
                    newState = (newHandValue, None, None)
                    result.append((newState, 1, newHandValue))
                # value exceeds threshold
                elif newHandValue > self.threshold:
                    newState = (newHandValue, None, None)
                    result.append((newState, 1, 0))
                # Peek card taken, game continued
                else:
                    newState = (newHandValue, None, tuple(list_newCardNums))
                    result.append((newState, 1, 0))
                return result

        # We peek, measure probability of which peek card we see, employ peek cost
        elif action == 'Peek':
            if peek_index is not None:
                result = []
                return result
            else:
                for i in range(len(self.cardValues)):
                    if probability[i] != 0:
                        newState = (handValue, i, numCards)
                        result.append((newState, probability[i], -self.peekCost))
            return result

        # If action is Quit, enter here, return results
        elif action == 'Quit':
            newState = (handValue, None, None)
            reward = handValue
            result.append((newState, 1, reward))
            return result

        return result

        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the
    optimal action at least 10% of the time. Have lots of low cards, but possibility of high car
    with low multiplicity. Peeking is good because you'd rather lose one cost but gain more than
    one from low card, but don't want to bust due to the 20 in the deck
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    cards = [2, 3, 4, 5, 20]
    multiplicity = 2
    threshold = 20
    peekingcost = 1

    MDP = BlackjackMDP(cards, multiplicity, threshold, peekingcost)
    return MDP
    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        # Terminal state case
        if newState is None:
            return

        # Get q_value for action
        q_hat_opt = self.getQ(state, action)

        # Initialize q_values list to find maximum value
        q_values = []
        # find all of the values for q with new state, actions from new state, take v_opt as max of those
        for new_action in self.actions(newState):
            q_values.append(self.getQ(newState, new_action))
        v_hat_opt = max(q_values)

        # Update the feature vectors
        for feature, val in self.featureExtractor(state, action):
            self.weights[feature] += -self.getStepSize()*(q_hat_opt - (reward + self.discount*v_hat_opt))*val

        # END_YOUR_CODE

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

def simulate_QL_over_MDP(mdp, featureExtractor):
    # NOTE: adding more code to this function is totally optional, but it will probably be useful
    # to you as you work to answer question 4b (a written question on this assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate Q-learning on the MDP,
    # and then print some stats comparing the policies learned by these two approaches.
    # BEGIN_YOUR_CODE

    # Q_Learning Algorithm, create object from class
    Q_Learn = QLearningAlgorithm(mdp.actions, mdp.discount(), featureExtractor)

    # First call simulate function on mdp case, use Q_learn object created above
    util.simulate(mdp, rl=Q_Learn, numTrials=30000, verbose=False)

    # Solve the value iteration problem
    val_iter = util.ValueIteration()
    val_iter.solve(mdp)

    # Initialize counter and similar to count similar instances for each iteration
    counter = 0
    similar = 0
    # Set exploration probability to 0 once solution is reached
    Q_Learn.explorationProb = 0

    # Iterate through all the states, find what the optimum policy is for both, compare
    for state in val_iter.pi:
        counter += 1
        if Q_Learn.getAction(state) == val_iter.pi[state]:
            similar += 1

    print('{} of the {} states are different'.format(counter-similar,counter))

    # END_YOUR_CODE


############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in the deck.
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card is (1, 1, 0, 1)
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each face value (len(counts) features).
#       Note: only add these features if the deck is not None.
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state

    # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
    features = []

    indicator_act_total = (('totes', action, total), 1)
    features.append(indicator_act_total)

    if counts is not None:
        indicator_counts = []
        counter = 0
        for val in counts:
            if val == 0:
                features.append((('prezzy/absy', action, counter), 0))
            else:
                features.append((('prezzy/absy', action, counter), 1))
            counter += 1

    counter = 0
    if counts is not None:
        for val in counts:
            features.append((('countsdracula', action, val, 'Card #' + str(counter)), 1))
            counter += 1
    return features
    # END_YOUR_CODE

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

def compare_changed_MDP(original_mdp, modified_mdp, featureExtractor):
    # NOTE: as in 4b above, adding more code to this function is completely optional, but we've added
    # this partial function here to help you figure out the answer to 4d (a written question).
    # Consider adding some code here to simulate two different policies over the modified MDP
    # and compare the rewards generated by each.
    # BEGIN_YOUR_CODE

    # Solve the value iteration problem
    val_iter = util.ValueIteration()
    val_iter.solve(original_mdp)

    rl_fixed = util.FixedRLAlgorithm(val_iter.pi)
    reward = util.simulate(mdp=modified_mdp, rl=rl_fixed, numTrials=30000)

    # Take expected reward as the average of all of the trials? Not sure if correct
    expected_reward = float(sum(reward))/float(len(reward))
    print('Expected Reward for FixedRLA = {}'.format(expected_reward))

    rl = QLearningAlgorithm(modified_mdp.actions, modified_mdp.discount(), featureExtractor)
    reward = util.simulate(mdp=modified_mdp, rl=rl, numTrials=30000)

    # Take expected reward as the average of all of the trials? Not sure if correct
    expected_reward = float(sum(reward))/float(len(reward))
    print('Expected Reward for New RLA = {}'.format(expected_reward))
    # END_YOUR_CODE

