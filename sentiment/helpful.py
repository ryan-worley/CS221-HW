import collections
import math

############################################################
# Problem 3a

def findAlphabeticallyLastWord(text):
    """
    Given a string |text|, return the word in |text| that comes last
    alphabetically (that is, the word that would appear last in a dictionary).
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find max() and list comprehensions handy here.
    """

    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    lst = sorted(text.split())
    return lst[-1]
    # END_YOUR_CODE

############################################################
# Problem 3b

def euclideanDistance(loc1, loc2):
    """
    Return the Euclidean distance between two locations, where the locations
    are pairs of numbers (e.g., (3, 5)).
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return math.sqrt((loc1[0]-loc2[0])**2+(loc1[1]-loc2[1])**2)
    # END_YOUR_CODE

############################################################
# Problem 3c

def mutateSentences(sentence):
    """
    Given a sentence (sequence of words), return a list of all "similar"
    sentences.
    We define a sentence to be similar to the original sentence if
      - it as the same number of words, and
      - each pair of adjacent words in the new sentence also occurs in the original sentence
        (the words within each pair should appear in the same order in the output sentence
         as they did in the original sentence.)
    Notes:
      - The order of the sentences you output doesn't matter.
      - You must not output duplicates.
      - Your generated sentence can use a word in the original sentence more than
        once.
    Example:
      - Input: 'the cat and the mouse'
      - Output: ['and the cat and the', 'the cat and the mouse', 'the cat and the cat', 'cat and the cat and']
                (reordered versions of this list are allowed)

    >>> mutateSentences('the cat and the mouse') == sorted(['and the cat and the', 'the cat and the mouse','the cat and the cat', 'cat and the cat and'])
        True
    """
    # BEGIN_YOUR_CODE (our solution is 20 lines of code, but don't worry if you deviate from this)

    order = collections.defaultdict(list)
    final = []
    words = sentence.split()

    # Create all dict to orgaize following words from original
    for z in range(len(words)-1):
        order[words[z]].append(words[z+1])

    # Makes starting word different every time
    for i in range(len(words)-1):
        store = []
        first = words[i]
        store.append(first)
        # Add number of words needed to make word string
        for j in range(len(words)-1):
            # For all current strings in the store matrix, find out that last word
            for l in range(len(store)):
                current_sent = store[l]
                wording = store[l].split()
                current_word = wording[-1]

                if current_word not in order.keys() or len(store[l]) <= j:
                    continue

                else:
                    # Iterate through all options for adding new words to a certain previous word
                    for k in range(len(order[current_word])):
                        new_sent = current_sent + ' ' + order[current_word][k]
                        store.append(new_sent)

        # Add all sentences in store that are same length and unique to final list
        for s in store:
            if s not in final and len(s.split()) == len(words):
                final.append(s)
    return sorted(final)
    # END_YOUR_CODE

############################################################
# Problem 3d

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as collections.defaultdict(float), return
    their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    z = 0
    for key in v1:
        if key in v2:
            z += v1[key] * v2[key]
    return z
    # END_YOUR_CODE

############################################################
# Problem 3e

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    for key in v1 and v2:
        if key in v2:
            v1[key] += scale*v2[key]
        else:
            v1[key] = scale*v2[key]
    return v1
    # END_YOUR_CODE

############################################################
# Problem 3f
def findSingletonWords(text):
    """
    Splits the string |text| by whitespace and returns the set of words that
    occur exactly once.
    You might find it useful to use collections.defaultdict(int).
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    lst = []
    d = collections.defaultdict(int)

    # Split words at whitespace, create dict of words
    for word in text.split():
        d[word] += 1

    # Word word in dict,
    for word in d:
        if d[word] == 1:
            lst.append(word)

    return set(lst)
    # END_YOUR_CODE

############################################################
# Problem 3g

def computeLongestPalindromeLength(text):
    """
    A palindrome is a string that is equal to its reverse (e.g., 'ana').
    Compute the length of the longest palindrome that can be obtained by deleting
    letters from |text|.
    For example: the longest palindrome in 'animal' is 'ama'.
    Your algorithm should run in O(len(text)^2) time.
    You should first define a recurrence before you start coding.

    """
    # BEGIN_YOUR_CODE (our solution is 19 lines of code, but don't worry if you deviate from this)
    # Create cache to store like results
    cache = {}

    # Create recursive function
    def getlength(string):
        """
        Function returns length of longest palindrome by recursing on
         itself. Takes a string input (originally full string) but shortens
         each iteration depending on the case that is run
         """
        # If the string is in the cache and stored already, will avoid running
        # code and instead will reset system to situation stored in cache
        if string in cache:
            return cache[string]

        # If the string is empty, show that length is 0
        if string == '':
            leng = 0

        # If the length of the string is 1, then this letter will be in middle
        # of palindrome, adding 1 to the length
        elif len(string) == 1:
            leng = 1

        # If the first and last letter are the same, add 2 to length
        elif string[0] == string[-1]:
            leng = 2 + getlength(string[1:-1])

        # If first and last letter are not the same, two situations advancing from
        # right and left
        else:
            if len(string) == 1:
                leng = 1
            left = getlength(string[1:])
            right = getlength(string[:-1])
            leng = max(left, right)
        cache[string] = leng
        return leng

    return getlength(text)
    # END_YOUR_CODE