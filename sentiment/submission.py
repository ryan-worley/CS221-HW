#!/usr/bin/python

import random
import collections
import math
import sys
import util
import string



############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    phi = collections.defaultdict(int)
    x = x.translate(None, string.punctuation)
    for word in x.split():
        phi[word] += 1
    return phi
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent


def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    import util
    # Create Predictor Function, Linear Predictor
    def predictor(z):
        phi = featureExtractor(z)
        num = util.dotProduct(phi, weights)
        if num > 0:
            return 1
        else:
            return -1

    weights = collections.defaultdict(float)  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)

    # Pull out the x and y value tuples from the training examples
    x = [j[0] for j in trainExamples]
    y = [l[1] for l in trainExamples]

    # Create Gradient Function to calculate gradient of hinge loss
    def dF(index, w):
        grad = collections.defaultdict(float)
        ex, why = trainExamples[index]
        phi = featureExtractor(ex)
        if (util.dotProduct(w, phi)) * why < 1:
            for k in phi.keys():
                grad[k] = -phi[k] * why
        else:
            for k in phi.keys():
                grad[k] = 0
        return grad

    # Create Iteration to run through gradient descent for number of iterations specified
    for t in range(numIters):
        for i in range(len(x)):
            gradient = dF(i, weights)
            for key in gradient.keys():
                weights[key] = weights[key] - eta/math.sqrt(1)*gradient[key]
        train_error = util.evaluatePredictor(trainExamples, predictor)
        test_error = util.evaluatePredictor(testExamples, predictor)
        print('Train Error = {}, Test Error = {}'.format(train_error, test_error))

    # END_YOUR_CODE
    return weights


############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.

    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = collections.defaultdict(float)
        for i in range(numExamples):
            x = random.choice(weights.keys())
            phi[x] = random.choice([-1, 1])*random.random()
        if util.dotProduct(phi,weights) > 0:
            y = 1
        else:
            y = 0
        # END_YOUR_CODE
        return (phi,y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        words = collections.defaultdict(int)
        no_spaces = x.replace(' ', '')
        if len(no_spaces)<n:
            return words
        for i in range(len(no_spaces)-n+1):
            words[no_spaces[i:i+n]] += 1
        return words
        # END_YOUR_CODE

    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    # Precompute part of vectorized squared loss function with data points
    precompute = collections.defaultdict(float)
    for counter in range(len(examples)):
        precompute[counter] = util.dotProduct(examples[counter], examples[counter])

    # Assign random centroids
    centers = random.sample(examples, K)

    # Loop through until number of iterations has been satisfied
    tolerance = ''
    TL = float('inf')
    TLnew = 0
    iters = 1

    while iters <= maxIters and tolerance != 'flag':
        # Check distance from centroids
        assignment = collections.defaultdict(int)
        center_count = []
        precomputed_centers = []
        center_increment = []
        n = 0

        for z in range(K):
            precomputed_centers.append(util.dotProduct(centers[z], centers[z]))

        for n in range(len(examples)):
            min_distance = float('inf')
            for z in range(K):
                distance = precompute[n] - 2*util.dotProduct(centers[z], examples[n]) + precomputed_centers[z]
                if distance <= min_distance:
                    min_index = z
                    min_distance = distance
            assignment[n] = min_index

        # Count the number of points associated with each center
        for i in range(K):
            center_count.append(list(assignment.values()).count(i))
            if center_count[i] == 0:
                center_increment.append(0)
            else:
                center_increment.append(1/float(center_count[i]))

        # Assign new centroids
        new_centers = {}
        for c in range(K):
            new_centers[c] = {}
            for pt in range(len(examples)):
                if assignment[pt] == c:
                    util.increment(new_centers[c], center_increment[c], examples[pt])

        centers = list(new_centers.values())

        p_train_loss = []
        for z in range(K):
            p_train_loss.append(util.dotProduct(centers[z], centers[z]))

        for x in range(len(examples)):
            TLnew += precompute[x] - 2*util.dotProduct(centers[assignment[x]], examples[x])+p_train_loss[assignment[x]]

        print('Training loss = {} on iteration {}'.format(TLnew, iters))

        if TLnew == TL:
            tolerance = 'flag'

        TL = TLnew
        TLnew = 0
        iters += 1

    return centers, assignment, TL
    # END_YOUR_CODE
