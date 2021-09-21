import nltk
import numpy as np
nltk.download('brown')
nltk.download('universal_tagset')
train_data = nltk.corpus.brown.tagged_sents(tagset='universal')[:10000]


# We create several functions that make it easier to create our required matrices 
def unique_tags(train_data):                                            # This functions simply creates an ordered list of unique tags that occur throughout the
    lst_of_tags = []                                                        # corpus
    for i in train_data:
        for j in i:
            if j[1] in lst_of_tags:
                pass
            else:
                lst_of_tags.append(j[1])
    return lst_of_tags

def unique_words(train_data):                                               # This creates an ordered list of unique words throughout the corpus
    lst_of_words = []
    for i in train_data:
        for j in i:
            if j[0] in lst_of_words:
                pass
            else:
                lst_of_words.append(j[0])
    lst_of_words.append('<UNK>')                                            # here we add a tag to represent unknown words or words that occur in the test data
    return lst_of_words                                                             # but are not in the training data
    
def nonunique_tags(train_data):
    lst_of_tags_nounique = []
    for i in train_data:
        for j in i:
            lst_of_tags_nounique.append(j[1])                               # returns a list of all the tags throughout the corpus wether unique ir non unique
    return lst_of_tags_nounique                                                         

def nonunique_words(train_data):
    lst_of_words_nounique = []                                                   # Creates a list of non unique words throughout the corpus 
    for i in train_data:
        for j in i:
            lst_of_words_nounique.append(j[0])
    return lst_of_words_nounique

def pairs(train_data):
    lst_of_tags_nounique = nonunique_tags(train_data)
    lst_of_bigrams_tuple = []
    for i in range(1, len(lst_of_tags_nounique)):                                         # this creates list of pair of tuples that occur successively in the data
        a = tuple([lst_of_tags_nounique[i-1],lst_of_tags_nounique[i]])                          # this helps in calculating transition probabilities 
        lst_of_bigrams_tuple.append(a)
    return lst_of_bigrams_tuple

def count_tuple(train_data):
    dic = {}
    for i in train_data:
        for j in i:
            if j not in dic.keys():                                                       # returns a dictionary  with tuples of word and tags as keys and  
                dic[j] = 1                                                                    # their counts as values that occur throughout the corpus
            else:
                dic[j] = dic[j] +1
    return dic

def count_tags(train_data):
    dic = {}
    for i in train_data:
        for j in i:                                                                         # returns a dictionary with tags as keys and counts as values 
            if j[1] not in dic.keys():                                                              # that occur throughout the corpus
                dic[j[1]] = 1
            else:
                dic[j[1]] = dic[j[1]] +1
    return dic

# Question 1 begins here 

def transitions(train_data):
    lst_of_tags = unique_tags(train_data)
    lst_of_bigrams_tuple = pairs(train_data)
    lst_of_tags_nounique = nonunique_words(train_data)
    transition_matrix = np.zeros((len(lst_of_tags),len(lst_of_tags)))
    for i in range(len(lst_of_tags)):                                                       # creates a matrix of transition probabilities. uses the list of unique 
        for j in range(len(lst_of_tags)):                                                           # tags. As that list is ordered the order of tags is preserved
            tup = tuple([lst_of_tags[i],lst_of_tags[j]])                                    
            num = lst_of_bigrams_tuple.count(tup) + 1                                       # we add one here and len(lst_of_tags) to the denominator to ensure 
            den = lst_of_tags_nounique.count(lst_of_tags[i]) + len(lst_of_tags)                     # smoothing
            prob = num/den
            transition_matrix[i,j] = prob
    return transition_matrix


def observation(train_data):
    lst_of_tags = unique_tags(train_data)
    lst_of_words = unique_words(train_data) 
    tags_dict = count_tags(train_data)
    tup_dict = count_tuple(train_data)
    observation_matrix = np.zeros((len(lst_of_tags), len(lst_of_words)))                                # we create observation matrix with tags as rows and words 
    for i in range(len(lst_of_tags)):                                                                       # as columns. We use the functions that create list of 
        for j in range(len(lst_of_words)):                                                              # unique words and tags to preserve order. 
            tup = tuple([lst_of_words[j], lst_of_tags[i]])                                  
            if tup in tup_dict.keys():
                num = tup_dict[tup] +1
            else:
                num = 1
            den = tags_dict[lst_of_tags[i]] + len(lst_of_words)                                                 # we ensure smoothing here in our function
            prob = num/den
            observation_matrix[i,j] = prob
    return observation_matrix


def initiation_probabilities(train_data):
    lst_of_tags = unique_tags(train_data)
    initial_probabilities = np.ones((len(lst_of_tags)))
    for i in range(len(lst_of_tags)):                                                                       # 1D array of initial probabilities of tags 
        for j in range(len(train_data)):
            if train_data[j][0][1] == lst_of_tags[i]:
                initial_probabilities[i] += 1                                                               # we ensure smoothing in our function by creating an 
                                                                                                        # of ones and then normalizing the array
    initial_probabilities = initial_probabilities/(len(train_data)+len(lst_of_tags))
    return initial_probabilities
    