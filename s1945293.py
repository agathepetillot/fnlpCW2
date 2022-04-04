import math

import nltk, inspect, sys, hashlib, itertools

import numpy as np

from nltk.corpus import brown

# module for computing a Conditional Frequency Distribution
from nltk.probability import ConditionalFreqDist

# module for computing a Conditional Probability Distribution
from nltk.probability import ConditionalProbDist, LidstoneProbDist

from nltk.tag import map_tag

from adrive2 import trim_and_warn

assert map_tag('brown', 'universal', 'NR-TL') == 'NOUN', '''
Brown-to-Universal POS tag map is out of date.'''


class HMM:
    def __init__(self, train_data):
        """
        Initialise a new instance of the HMM.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        """
        self.train_data = train_data

        # Emission and transition probability distributions
        self.emission_PD = None
        self.transition_PD = None
        self.states = []
        self.viterbi = []
        self.backpointer = []

    # Q1

    # Compute emission model using ConditionalProbDist with a LidstoneProbDist estimator.
    #   To achieve the latter, pass a function
    #    as the probdist_factory argument to ConditionalProbDist.
    #   This function should take 3 arguments
    #    and return a LidstoneProbDist initialised with +0.001 as gamma and an extra bin.
    #   See the documentation/help for ConditionalProbDist to see what arguments the
    #    probdist_factory function is called with.
    def emission_model(self, train_data):
        """Compute an emission model based on labelled training data.
        Don't forget to lowercase the observation otherwise it mismatches the test data.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The emission probability distribution and a list of the states
        :rtype: Tuple[ConditionalProbDist, list(str)]
        """

        # Don't forget to lowercase the observation otherwise it mismatches the test data
        # Do NOT add <s> or </s> to the input sentences'
        data = []

        # switch observation(word) and tag in each tuple and add to data list
        # lowercase observation(word)
        for filter_data in train_data:
            data_list = [(tag, word.lower()) for (word, tag) in filter_data]
            data.extend(data_list)

        # calculate emission prob
        emission_FD = ConditionalFreqDist(data)
        estimator = lambda fd: LidstoneProbDist(fd, 0.001, fd.B() + 1)
        self.emission_PD = ConditionalProbDist(emission_FD, estimator)
        self.states = emission_FD.conditions()

        return self.emission_PD, self.states

    # Q1

    # Access function for testing the emission model
    # For example model.elprob('VERB','is') might be -1.4
    def elprob(self, state, word):
        """
        The log of the estimated probability of emitting a word from a state

        :param state: the state name
        :type state: str
        :param word: the word
        :type word: str
        :return: log base 2 of the estimated emission probability
        :rtype: float
        """

        return self.emission_PD[state].logprob(word)

    # Q2
    # Compute transition model using ConditionalProbDist with the same
    # estimator as above (but without the extra bin)
    # See comments for emission_model above for details on the estimator.
    def transition_model(self, train_data):
        """
        Compute a transition model using a ConditionalProbDist based on
          labelled data.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The transition probability distribution
        :rtype: ConditionalProbDist
        """

        # TODO: prepare the data

        # The data object should be an array of tuples of conditions and observations,
        # in our case the tuples will be of the form (tag_(i),tag_(i+1)).
        # DON'T FORGET TO ADD THE START SYMBOL </s> and the END SYMBOL </s>

        data = []
        # adding start and end tokens to each sentence
        for s in train_data:
            data.append([("<s>", "<s>")] + s + [("</s>", "</s>")])

        # generate tags
        tagGenerators = (((s[i][1], s[i + 1][1]) for i in range(len(s) - 1)) for s in data)
        dataTags = itertools.chain.from_iterable(tagGenerators)

        # calculate transition prob
        transition_FD = ConditionalFreqDist(dataTags)
        estimator = lambda fd: LidstoneProbDist(fd, 0.001)
        self.transition_PD = ConditionalProbDist(transition_FD, estimator)

        return self.transition_PD

    # Q2
    # Access function for testing the transition model
    # For example model.tlprob('VERB','VERB') might be -2.4
    def tlprob(self, state1, state2):
        """
        The log of the estimated probability of a transition from one state to another

        :param state1: the first state name
        :type state1: str
        :param state2: the second state name
        :type state2: str
        :return: log base 2 of the estimated transition probability
        :rtype: float
        """
        return self.transition_PD[state1].logprob(state2)

    # Train the HMM
    def train(self):
        """
        Trains the HMM from the training data
        """
        self.emission_model(self.train_data)
        self.transition_model(self.train_data)

    # Part B: Implementing the Viterbi algorithm.

    # Q3
    # Initialise data structures for tagging a new sentence.
    # Describe the data structures with comments.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD
    # Input: first word in the sentence to tag and the total number of observations.
    def initialise(self, observation, number_of_observations):
        """
        Initialise data structures self.viterbi and self.backpointer for tagging a new sentence.

        :param observation: the first word in the sentence to tag
        :type observation: str
        :param number_of_observations: the number of observations
        :type number_of_observations: int
        """
        # pseudocode for initialisation:
        # for each state s from 1 to N do; initialization step
        # viterbi[s, 1]←πs ∗ bs(o1)
        # backpointer[s, 1]←0

        # Initialise step 0 of viterbi, including
        #  transition from <s> to observation
        # use costs (- log-base-2 probabilities)

        # A 2D list is used at each point to manage the possible states
        self.viterbi = [[0]*number_of_observations for _ in range(len(self.states))]
        # Denotes the sequence of tags in a 2D list
        self.backpointer = [[0]*number_of_observations for _ in range(len(self.states))]

        for s in self.states:
            # tl = self.tlprob("<s>", self.states[s])
            # el = self.elprob(self.states[s], observation)

            # initialise for transition from <s>, beginning of sentence
            self.viterbi[self.states.index(s)][0] = -self.tlprob("<s>", s) - self.elprob(s, observation)

    # Access function for testing the viterbi data structure
    # For example model.get_viterbi_value('VERB',2) might be 6.42
    def get_viterbi_value(self, state, step):
        """
        Return the current value from self.viterbi for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step:  if negative,
          counting backwards from the end, i.e. -1 means the last step
        :type step: int
        :return: The value (a cost) for state as of step
        :rtype: float
        """

        return self.viterbi[self.states.index(state)][step]

    # Q3
    # Access function for testing the backpointer data structure
    # For example model.get_backpointer_value('VERB',2) might be 'NOUN'
    def get_backpointer_value(self, state, step):
        """
        Return the current backpointer from self.backpointer for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step:  if negative,
          counting backwards from the end, i.e. -1 means the last step
        :type step: int
        :return: The state name to go back to at step-1
        :rtype: str
        """

        return self.states[self.backpointer[self.states.index(state)][step]]

    # Q4a
    # Tag a new sentence using the trained model and already initialised data structures.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD.
    # Update the self.viterbi and self.backpointer data structures.
    # Describe your implementation with comments.
    def tag(self, observations):
        """
        Tag a new sentence using the trained model and already initialised data structures.

        :param observations: List of words (a sentence) to be tagged
        :type observations: list(str)
        :return: List of tags corresponding to each word of the input
        """

        # for each time step t from 2 to T do; recursion step
        # for each state s from 1 to N do
        # viterbi[s, t]←...
        # backpointer[s, t]←

        tags = []

        T = len(observations)

        for t in range(1, T):
            for s in self.states:
                self.viterbi[self.states.index(s)][t] = min([(self.viterbi[self.states.index(sDash)][t - 1] - self.tlprob(sDash, s) -
                                                                self.elprob(s, observations[t])) for sDash in self.states])
                self.backpointer[self.states.index(s)][t] = np.argmin([(self.viterbi[self.states.index(sDash)][t - 1] -
                                                                        self.tlprob(sDash, s) - self.elprob(s, observations[t])) for sDash in self.states])

        # TODO
        # Add a termination step with cost based solely on cost of transition to </s> , end of sentence.

        # TODO
        # Reconstruct the tag sequence using the backpointers.
        # Return the tag sequence corresponding to the best path as a list.
        # The order should match that of the words in the sentence.

        (_, last_pointer) = min([(self.viterbi[self.states.index(s)][T - 1] - self.tlprob(s, '</s>'), self.states.index(s)) for s in self.states])
        tags.append(self.states[last_pointer])

        for t in range(T, -1, -1):
            if t != 1 and t != 0:
                next_state = self.states[self.backpointer[last_pointer][t - 1]]
                tags.append(next_state)
                last_pointer = self.backpointer[last_pointer][t - 1]

        # reverse tags as we start at the end of the sentence
        tags = tags[::-1]

        return tags

    def tag_sentence(self, sentence):
        """
        Initialise the HMM, lower case and tag a sentence. Returns a list of tags.
        :param sentence: the sentence
        :type sentence: list(str)
        :rtype: list(str)
        """

        # Create instance of HMM class and initialise the training set.
        # Train the HMM.
        sentence_lower = [word.lower() for word in sentence]
        self.initialise(sentence_lower[0], len(sentence_lower))
        tags = self.tag(sentence_lower)

        return tags

def answer_question4b():
    """
    Report a hand-chosen tagged sequence that is incorrect, correct it
    and discuss
    :rtype: list(tuple(str,str)), list(tuple(str,str)), str
    :return: incorrectly tagged sequence, correctly tagged sequence and your answer [max 280 chars]
    """

    # One sentence, i.e. a list of word/tag pairs, in two versions
    #  1) As tagged by your HMM
    #  2) With wrong tags corrected by hand
    tagged_sequence = [('For', 'ADP'), ('a', 'DET'), ('richer', 'NOUN'), (',', '.'), ('fuller', 'CONJ'), ('life', 'NOUN'), (',', '.'), ('read', 'VERB')]
    correct_sequence = [('For', 'ADP'), ('a', 'DET'), ('richer', 'ADJ'), (',', '.'), ('fuller', 'ADJ'), ('life', 'NOUN'), (',', '.'), ('read', 'VERB')]
    # Why do you think the tagger tagged this example incorrectly?
    answer = inspect.cleandoc("""2 words were tagged incorrectly, richer and fuller. The tagger thought fuller was a CONJ as
    CONJs are often in between commas so the elprob was much greater for CONJ for fuller. Subsequently this tagged richer 
    incorrectly (NOUN) meaning a wealthier people as a group.""")

    return tagged_sequence, correct_sequence, trim_and_warn("Q4a", 280, answer)


# Q5a
def hard_em(labeled_data, unlabeled_data, k):
    """
    Run k iterations of hard EM on the labeled and unlabeled data.
    Follow the pseudo-code in the coursework instructions.

    :param labeled_data:
    :param unlabeled_data:
    :param k: number of iterations
    :type k: int
    :return: HMM model trained with hard EM.
    :rtype: HMM
    """
    raise NotImplementedError()
    # T_0 = HMM(labeled_data)
    # for i in range(k-1):
    return ... # fix me


def answer_question5b():
    """
    Sentence:  In    fact  he    seemed   delighted  to  get   rid  of  them   .
    Gold POS:  ADP   NOUN  PRON  VERB     VERB      PRT  VERB  ADJ  ADP  PRON  .
    T_0     :  PRON  VERB  NUM    ADP     ADJ       PRT  VERB  NUM  ADP  PRON  .
    T_k     :  PRON  VERB  PRON  VERB     ADJ       PRT  VERB  NUM  ADP  NOUN  .

    1) T_0 erroneously tagged "he" as "NUM" and T_k correctly identifies it as "PRON".
        Speculate why additional unlabeled data might have helped in that case.
        Refer to the training data (inspect the 20 sentences!).
    2) Where does T_k mislabel a word but T_0 is correct? Why do you think did hard EM hurt in that case?

    :rtype: str
    :return: your answer [max 500 chars]
    """

    return trim_and_warn("Q5b", 500, inspect.cleandoc("""1)Hard EM as a way to bootstrap better parameter values when we don’t
    have enough annotated data. Hence since we had so much unlabeled data, EM worked more efficiently whereas T_0 was unable to 
    identify correct values.
    2)T_k mislabeled them as a NOUN whereas T_0 labeled it correctly (PRON). 
    The likelihood function has multiple local optima. EM will converge to one of these however, hard EM won't necessarily. Thus
    this may have causes it to incorrectly label T_k."""))

def answer_question6():
    """
    Suppose you have a hand-crafted grammar that has 100% coverage on
        constructions but less than 100% lexical coverage.
        How could you use a POS tagger to ensure that the grammar
        produces a parse for any well-formed sentence,
        even when it doesn't recognise the words within that sentence?

    :rtype: str
    :return: your answer [max 500 chars]
    """

    return trim_and_warn("Q6", 500, inspect.cleandoc("""
    It can be used to tag words that aren't by the hand-crafted lexicon. 
    Doing so, any sentence will be fully tagged and can be parsed by the 
    hand-crafted grammar. For example, if we have an unknown tag, the POS tagger 
    replace this unknown tag with a known one (e.g. NOUN). This is because 
    it will create a constituency tree which allows us to find the unknown tag. 
    This is expected to do better since the other parser is unable to parse sentences
    with missing tags.
    """))


def answer_question7():
    """
    Why else, besides the speedup already mentioned above, do you think we
    converted the original Brown Corpus tagset to the Universal tagset?
    What do you predict would happen if we hadn't done that?  Why?

    :rtype: str
    :return: your answer [max 500 chars]
    """

    return trim_and_warn("Q7", 500, inspect.cleandoc("""\
    The Universal tagset is a small, coarse-grained tagset. A smaller tagset gives higher accuracy.
    However, a tiny tagset tends to make the tagging system less useful as it provides less information.
    Hence, the best results seem to be when the tagset size is between 15 and 25. The Universal tagset 
    contains 17 tags whereas the brown corpus tagset is vastly larger. So, if we were
    to have used the brown tagset, the accuracy of the POS tagging would have been significantly lower."""))


def compute_acc(hmm, test_data, print_mistakes):
    """
    Computes accuracy (0.0 - 1.0) of model on some data.
    :param hmm: the HMM
    :type hmm: HMM
    :param test_data: the data to compute accuracy on.
    :type test_data: list(list(tuple(str, str)))
    :param print_mistakes: whether to print the first 10 model mistakes
    :type print_mistakes: bool
    :return: float
    """

    # TODO: modify this to print the first 10 sentences with at least one mistake if print_mistakes = True
    all_mistake_sent = []
    correct = 0
    incorrect = 0
    for sentence in test_data:
        s = [word for (word, tag) in sentence]
        tags = hmm.tag_sentence(s)

        mistake_sent = []
        print_mistakes = False
        for ((word, gold), tag) in zip(sentence, tags):
            if tag == gold:
                correct += 1
            else:
                print_mistakes = True
                incorrect += 1
            mistake_sent.append(((word, gold), tag))

    if print_mistakes:
        all_mistake_sent.append(mistake_sent)

    # Save first 10 incorrectly tagged test sentences along with their correct version
    first_ten_mistakes = all_mistake_sent[:10]

    analyse = first_ten_mistakes[:1]
    print(analyse)

    return float(correct) / (correct + incorrect)

# Useful for testing
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # http://stackoverflow.com/a/33024979
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def answers():
    global tagged_sentences_universal, test_data_universal, \
        train_data_universal, model, test_size, train_size, ttags, \
        correct, incorrect, accuracy, \
        good_tags, bad_tags, answer4b, answer5, answer6, answer7, answer5b, \
        t0_acc, tk_acc

    # Load the Brown corpus with the Universal tag set.
    tagged_sentences_universal = brown.tagged_sents(categories='news', tagset='universal')

    # Divide corpus into train and test data.
    test_size = 500
    train_size = len(tagged_sentences_universal) - test_size

    # tail test set
    test_data_universal = tagged_sentences_universal[-test_size:]  # [:test_size]
    train_data_universal = tagged_sentences_universal[:train_size]  # [test_size:]
    if hashlib.md5(''.join(map(lambda x: x[0],
                               train_data_universal[0] + train_data_universal[-1] + test_data_universal[0] +
                               test_data_universal[-1])).encode(
            'utf-8')).hexdigest() != '164179b8e679e96b2d7ff7d360b75735':
        print('!!!test/train split (%s/%s) incorrect -- this should not happen, please contact a TA !!!' % (
        len(train_data_universal), len(test_data_universal)), file=sys.stderr)

    # Create instance of HMM class and initialise the training set.
    model = HMM(train_data_universal)

    # Train the HMM.
    model.train()

    # Some preliminary sanity checks
    # Use these as a model for other checks
    e_sample = model.elprob('VERB', 'is')
    if not (type(e_sample) == float and e_sample <= 0.0):
        print('elprob value (%s) must be a log probability' % e_sample, file=sys.stderr)

    t_sample = model.tlprob('VERB', 'VERB')
    if not (type(t_sample) == float and t_sample <= 0.0):
        print('tlprob value (%s) must be a log probability' % t_sample, file=sys.stderr)

    if not (type(model.states) == list and \
            len(model.states) > 0 and \
            type(model.states[0]) == str):
        print('model.states value (%s) must be a non-empty list of strings' % model.states, file=sys.stderr)

    print('states: %s\n' % model.states)

    ######
    # Try the model, and test its accuracy [won't do anything useful
    #  until you've filled in the tag method
    ######
    s = 'the cat in the hat came back'.split()
    ttags = model.tag_sentence(s)
    print("Tagged a trial sentence:\n  %s" % list(zip(s, ttags)))

    v_sample = model.get_viterbi_value('VERB', 5)
    if not (type(v_sample) == float and 0.0 <= v_sample):
        print('viterbi value (%s) must be a cost' % v_sample, file=sys.stderr)

    b_sample = model.get_backpointer_value('VERB', 5)
    if not (type(b_sample) == str and b_sample in model.states):
        print('backpointer value (%s) must be a state name' % b_sample, file=sys.stderr)

    # check the model's accuracy (% correct) using the test set
    accuracy = compute_acc(model, test_data_universal, print_mistakes=True)
    print('\nTagging accuracy for test set of %s sentences: %.4f' % (test_size, accuracy))

    #Tag the sentence again to put the results in memory for automarker.
    model.tag_sentence(s)

    # Question 5a
    # Set aside the first 20 sentences of the training set
    num_sentences = 20
    semi_supervised_labeled = train_data_universal[:num_sentences]  # type list(list(tuple(str, str)))
    semi_supervised_unlabeled = [[word for (word, tag) in sent] for sent in train_data_universal[num_sentences:]]  # type list(list(str))
    print("Running hard EM for Q5a. This may take a while...")
    t0 = hard_em(semi_supervised_labeled, semi_supervised_unlabeled, 0) # 0 iterations
    tk = hard_em(semi_supervised_labeled, semi_supervised_unlabeled, 3)
    print("done.")

    t0_acc = compute_acc(t0, test_data_universal, print_mistakes=False)
    tk_acc = compute_acc(tk, test_data_universal, print_mistakes=False)
    print('\nTagging accuracy of T_0: %.4f' % (t0_acc))
    print('\nTagging accuracy of T_k: %.4f' % (tk_acc))
    ########

    # Print answers for 4b, 5b, 6 and 7.
    bad_tags, good_tags, answer4b = answer_question4b()
    print('\nA tagged-by-your-model version of a sentence:')
    print(bad_tags)
    print('The tagged version of this sentence from the corpus:')
    print(good_tags)
    print('\nDiscussion of the difference:')
    print(answer4b)
    answer5b = answer_question5b()
    print("\nFor Q5b:")
    print(answer5b)
    answer6 = answer_question6()
    print('\nFor Q6:')
    print(answer6)
    answer7 = answer_question7()
    print('\nFor Q7:')
    print(answer7)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--answers':
        import adrive2
        from autodrive_embed import run, carefulBind

        with open("userErrs.txt", "w") as errlog:
            run(globals(), answers, adrive2.a2answers, errlog)
    else:
        answers()
