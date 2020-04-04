import shell
import util
import wordsegUtil

############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        state = (0, 0)
        return state
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        return state == (len(self.query), len(self.query))
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        result = []
        # returns list of: current word, futureletters, cost
        (pi, ci) = state

        # Create a case where space not implemented
        if ci <= len(self.query) - 1:
            newState = (pi, ci+1)
            action = 'None'
            cost = 0
            result.append((action, newState, cost))

        # Create a case where space is implemented
        if ci <= len(self.query) - 1:
            newState = (ci+1, ci+1)
            action = ci + 1  # Index where to put space
            cost = self.unigramCost(self.query[pi:ci+1])
            result.append((action, newState, cost))

        return result
        # END_YOUR_CODE

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 10 lines of code, but don't worry if you deviate from this)
    spaces = []
    for x in ucs.actions:
        if x != 'None':
            spaces.append(x)
    spaces = list(enumerate(sorted(spaces)))

    sentence = ''
    for (count, index) in spaces:
        if count == 0:
            sentence += query[:index] + ' '
            previous = index
        else:
            sentence += query[previous:index] + ' '
            previous = index

    return sentence
    # END_YOUR_CODE

############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords # List of words split up here
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills # Returns possible fills of letters

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0, wordsegUtil.SENTENCE_BEGIN
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        index, word = state
        if index == len(self.queryWords):
            return True
        return False
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 16 lines of code, but don't worry if you deviate from this)
        result = []
        (index, previousWord) = state

        possibilities = self.possibleFills(self.queryWords[index])

        if len(possibilities) == 0:
            action = 'None'
            newState = (index + 1, self.queryWords[index])
            cost = self.bigramCost(previousWord, self.queryWords[index])
            result.append((action, newState, cost))

        else:
            for word in possibilities:
                action = word
                newState = (index + 1, word)
                cost = self.bigramCost(previousWord, word)
                result.append((action, newState, cost))

        return result
        # END_YOUR_CODE


def insertVowels(queryWords, bigramCost, possibleFills):

    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))

    return ' '.join(ucs.actions)
    # END_YOUR_CODE

############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        ci = 0  # current index
        si = 0  # start index
        state = (ci, si, wordsegUtil.SENTENCE_BEGIN)
        return state
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        ci, si, previousWord = state
        if ci == len(self.query) and si == len(self.query):  # Means ci, si have reached end
            return True
        return False
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 23 lines of code, but don't worry if you deviate from this)
        result = []
        ci, si, previousWord = state
        possibilities = self.possibleFills(self.query[si:ci+1])

        if ci <= len(self.query)-1:
            if len(possibilities) == 0:
                cost = 0
                action = 'None'
                result.append((action, (ci+1, si, previousWord), cost))

            else:
                no_cost = 0
                action = 'None'
                result.append((action, (ci+1, si, previousWord), no_cost))
                for word in possibilities:
                    cost = self.bigramCost(previousWord, word)
                    result.append((word, (ci+1, ci+1, word), cost))
        return result
        # END_YOUR_CODE

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 11 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))

    toPrint = []
    words = ucs.actions

    for word in words:
        if word != 'None':
            toPrint.append(word)

    return ' '.join(toPrint)
    # END_YOUR_CODE

############################################################

if __name__ == '__main__':
    shell.main()
