# Create functions that help with Viterbi function
import nltk
import numpy as np
from Q1 import transitions, observation, initiation_probabilities, unique_words, unique_tags



nltk.download('brown')
nltk.download('universal_tagset')
train_data = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]





def grab_index(word, train_data):
    lst_of_words = unique_words(train_data)
    if word in lst_of_words:
        return lst_of_words.index(word)
    else:                                                                           # takes the word and returns the index of that word in the ordered list of 
        return len(lst_of_words) - 1                                                     # unique words we have created before. If the word is not in the unique list 
                                                                                            # then the index of UNKNOWN is assigned to it 

def lst_of_ints(sentence, train_data):
    lst = []                                                                  # take a sentence and creates a list of index for each word in the same order
    for i in sentence:                                                              # these indexes correspond to the unique list of words created before
        lst.append(grab_index(i[0], train_data))
    return lst


lst_of_tags = unique_tags(train_data)

def viterbi(lst, pi, tran, obs):                              
    states_seq = []
    log_ini = np.log(pi)
    log_trans = np.log(tran)                                                      # the viterbi function 
    log_obs = np.log(obs)                                                 #convert matrices to log 
    path = np.zeros((log_trans.shape[1], len(lst)))
    path[:,0] = log_ini + log_obs[:,lst[0]]
    for j in range(1,path.shape[1]):                                         # we make use to numpy functions max and argmax functions
        for i in range(path.shape[0]):
            path[i,j] = max(path[:,j-1] + log_trans[:,i]) + log_obs[i,lst[j]]
    for i in range(path.shape[1]):
        states_seq.append(lst_of_tags[np.argmax(path[:,i])])
    return states_seq

# We have provided a function below and test case to measure the accuracy against the truth for our viterbi function. We use this function to report our results 
# in the pdf 
sentence = nltk.corpus.brown.tagged_sents(tagset='universal')[10152]
lst_of_tags = unique_tags(train_data)
train_data = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]
initial = initiation_probabilities(train_data)
obs = observation(train_data)
trans = transitions(train_data)

def measure_accuracy(sentence):
    lst1 = lst_of_ints(sentence,train_data)
    
    l = viterbi(lst1, initial, trans, obs)
    print(l)
    correct = 0
    for i in range(len(sentence)):
        if sentence[i][1] == l[i]:
            correct = correct + 1
    return correct/len(sentence)


print(measure_accuracy(sentence))