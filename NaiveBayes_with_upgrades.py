import pandas as pd
import math
import random
from nltk import ngrams

conditional_dict = {}
dict = {}
words_in_class = [0, 0, 0, 0]  # count of words in each class
train = None

"""
the create_dictionary function controls the creation of the dictionary.
By calling the "add_to_dictionary()",  "clean_data() & the conditional_dict() functions 

Parameters: Dataframe - This is the dataframe that will be used for training.

Return: None - helper function to build the global variables.

"""


def create_dictionary(DataFrame):

    for index, row in DataFrame.iterrows():
        add_to_dictionary(row["class"], clean_data(row["abstract"]))
    _conditional_dict()


"""
n_grams creates a n-gram for the given abstract. 
change secodn value of the (,) to adjust the number of values in the n-gram

"""


def n_gram(list):
    ngrams_ = ngrams(list, 1)  # change this value to change n- gram
    list = []
    for word in ngrams_:
        list.append(" ".join(word))
    return list


"""
The clean_data() does some preprocessesing such as removing the 10 most common words from the given abstract. 

Parameters: abstract - this is the training data abstract that will be 

Return: list of the words from the abstract. excluding the top 10 most common words and concationation of the words below.
"""


def clean_data(abstract):
    list = abstract.split(" ")
    # improvement to remove the uncorrilated words.
    for element in ['the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it']:
        while element in list:
            list.pop(list.index(element))

    for word in list:
        if word == "homo" and (list.index(word)+1 < len(list)):
            if (list[(list.index(word)+1)] == "sapiens"):
                list.pop(list.index(word)+1)
                word = "homo-sapiens"

        if word == "escherichia" and (list.index(word)+1 < len(list)):
            if (list[(list.index(word)+1)] == "coli"):
                list.pop(list.index(word)+1)
                word = "escherichia-coli"

        if word == "human" and (list.index(word)+2 < len(list)):
            if (list[(list.index(word)+1)] == "immunodeficiency") and (list[(list.index(word)+2)] == "virus"):
                list.pop(list.index(word)+1)
                word = "human-immunodeficiency-virus"

    return n_gram(list)


"""
The  add_to_dictionary() adds counts to the given dictionary key (word) based on the given class. 

Parameters: class - the assigned class of the abstract
            abstract - the abstract of the research paper.

Return:  None - it updates global variables "dict" & "words_in_class"
"""


def add_to_dictionary(_class, abstract):
    # print(list)
    for word in abstract:
        if word not in dict:  # adding words to dictionary
            if _class == "A":
                dict[word] = [1, 0, 0, 0]
                words_in_class[0] += 1
            if _class == "B":
                dict[word] = [0, 1, 0, 0]
                words_in_class[1] += 1
            if _class == "E":
                dict[word] = [0, 0, 1, 0]
                words_in_class[2] += 1
            if _class == "V":
                dict[word] = [0, 0, 0, 1]
                words_in_class[3] += 1
        else:  # appending count to existing words
            if _class == "A":
                dict[word][0] += 1
                words_in_class[0] += 1
            if _class == "B":
                dict[word][1] += 1
                words_in_class[1] += 1
            if _class == "E":
                dict[word][2] += 1
                words_in_class[2] += 1
            if _class == "V":
                dict[word][3] += 1
                words_in_class[3] += 1

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


"""
The  _conditional_dict creates the dictionary of conditional proabilities  

Parameters: uses global variable "dict"  

Return: None -  updates the global variable "conditional_dict"
"""


def _conditional_dict():
    global conditional_dict
    conditional_dict = {}
    for x in dict.keys():
        word_probs = conditional_proability_word(x)
        conditional_dict[x] = [word_probs[0],
                               word_probs[1], word_probs[2], word_probs[3]]

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


"""
The  conditional_proability_word calculates the a words given conditional proabilities

Parameters: word being caluclated

Return: probabilities -  the conditional probabilities of a word being in a given class.
"""


def conditional_proability_word(word):

    probabilities = [0, 0, 0, 0]
    if word not in dict:
        return probabilities

    for i in range(0, 4):
        # print(i)
        p = math.log((dict.get(word)[i]+1)/(words_in_class[i]+len(dict)), 2)
        probabilities[i] = p
    return probabilities


"""def conditional_proability_abstract(abstract):

    list = abstract.split(" ")

    A_list = []
    B_list = []
    E_list = []
    V_list = []

    for word in list:
        word_prob = conditional_proability_word(word)
        A_list.append(word_prob[0])
        B_list.append(word_prob[1])
        E_list.append(word_prob[2])
        V_list.append(word_prob[3])

    A = sum(A_list)
    B = sum(B_list)
    E = sum(E_list)
    V = sum(V_list)
    return A, B, E, V"""

"""
The  "prior_proability" function calculates the prior proability of a given abstract.

Parameters: takes the global variable of train which is the training set, and calculates the prior proabilities of it

Return: tuple of prior proabilities - for each given class 

"""


def prior_proability():
    global train
    A = math.log(train.value_counts("class")["A"] / len(train), 2)
    B = math.log(train.value_counts("class")["B"] / len(train), 2)
    E = math.log(train.value_counts("class")["E"] / len(train), 2)
    V = math.log(train.value_counts("class")["V"] / len(train), 2)
    return A, B, E, V


"""def posterior_proability_abstract(abstract):

    prior = prior_proability()
    conditional = conditional_proability_abstract(abstract)
    posterior_proabilities = []
    for i in range(0, 4):
        posterior_proabilities.append(prior[i] + conditional[i])
    # for i in range(0,4):
        # posterior_proabilities[i] = pow(2,posterior_proabilities[i])

    if posterior_proabilities.index(max(posterior_proabilities)) == 0:
        return "A"
    if posterior_proabilities.index(max(posterior_proabilities)) == 1:
        return "B"
    if posterior_proabilities.index(max(posterior_proabilities)) == 2:
        return "E"
    if posterior_proabilities.index(max(posterior_proabilities)) == 3:
        return "V"
"""
"""
The  "posterior_proability_abstract()" calulates the posterior proability of the given abstract. By calling the "prior_proability()" and "conditional_proability_abstract()" 
and sum them

Parameters: abstract

Return: The most likly class for the given abstract. (A,B,E,V)

"""


def posterior_proability_abstract(abstract):
    prior = prior_proability()
    conditional = conditional_proability_abstract(abstract)
    posterior_proabilities = []
    for i in range(0, 4):
        posterior_proabilities.append(prior[i] + conditional[i])

    if posterior_proabilities.index(max(posterior_proabilities)) == 0:
        return "A"
    if posterior_proabilities.index(max(posterior_proabilities)) == 1:
        return "B"
    if posterior_proabilities.index(max(posterior_proabilities)) == 2:
        return "E"
    if posterior_proabilities.index(max(posterior_proabilities)) == 3:
        return "V"


"""
The  "conditional_proability_abstract()" calulates the conditional proability of the given abstract. 
By calulating the onditional proability for each word from the abstract and summing them.

Parameters: abstract

Return: conditional proability for the classes retunred as a tuple.

"""


def conditional_proability_abstract(abstract):
    global conditional_dict
    list = abstract.split(" ")
    list = n_gram(list)

    A_list = []
    B_list = []
    E_list = []
    V_list = []

    for word in list:
        if word in conditional_dict:
            A_list.append(conditional_dict[word][0])
            B_list.append(conditional_dict[word][1])
            E_list.append(conditional_dict[word][2])
            V_list.append(conditional_dict[word][3])

    A = sum(A_list)
    B = sum(B_list)
    E = sum(E_list)
    V = sum(V_list)
    return A, B, E, V

# -------------Validating-------------------------------------


DataFrame = pd.read_csv('trg.csv', sep=',')  # imports the training set.


"""
The  "validate_accuracy()" calulates the accuracy of our training stage on the validation data.

Parameters: the subset of the Dataframe that will be used to validate the training stage.

Return: the accuracy of the training stage on the validation data.

"""


def validate_accuracy(validate):

    true, false = 0, 0
    for index, row in validate.iterrows():
        if row["class"] == posterior_proability_abstract(row["abstract"]):
            true += 1
        else:
            false += 1
    return true/len(validate)


"""
The  "cross_validation()" sets up a cross validation enviroment

Parameters: uses the original imported Dataframe.

Return: accuracy of our algorithm using the cross validation techniqie. 

"""


def cross_validation():
    accuracy_list = []
    i = 0
    split_val = int(len(DataFrame)/20)  # 20 folds

    while i < len(DataFrame):
        train = DataFrame.copy()
        validate = train.iloc[i:i+split_val]

        train = train.drop(train.index[i:i+split_val])

        global dict, words_in_class
        dict = {}
        words_in_class = [0, 0, 0, 0]

        create_dictionary(train)
        accuracy_list.append(validate_accuracy(validate))

        i += split_val
    print(accuracy_list)

    return sum(accuracy_list)/len(accuracy_list)


"""
The  "train_validate_split()" sets up a 80/20 split enviroment 

Parameters: uses the original imported Dataframe.

Return: accuracy of our algorithm using the train-validate techniqie. 

"""


def train_validate_split():
    split_value = int(len(DataFrame)*0.8)
    global train
    train = DataFrame.iloc[:split_value]
    validate = DataFrame.iloc[split_value:]

    create_dictionary(train)
    # conditional_dict()
    # print(dict)
    return validate_accuracy(validate)

# print(train_test_split())


# --------------Testing-----------------------------------------------------------------
"""
This part is the controller to run the testing stage for our testing data.

Itterates through all the rows of the tst.csv file and creates a new prediction for it. 

It creates a csv file to be imported to Kaggle.
"""


def testing():
    global train
    train = DataFrame
    create_dictionary(DataFrame)  # using full training set.

    list = []
    Test_Dataframe = pd.read_csv('tst.csv', sep=',')

    for index, row in Test_Dataframe.iterrows():
        class_val = posterior_proability_abstract(row["abstract"])
        list.append([row["id"], class_val])

    Return_Dataframe = pd.DataFrame(list, columns=["id", "class"])
    # print(Return_Dataframe)

    # VincentVanDerVegteTest-unigram-log-removal-test-validate-8020.csv
    Return_Dataframe.to_csv('vvan312.csv', index=False)


def random_over_sampling():

    A = DataFrame.value_counts("class")["A"]
    B = DataFrame.value_counts("class")["B"]
    E = DataFrame.value_counts("class")["E"]
    V = DataFrame.value_counts("class")["V"]

    grouped = DataFrame.groupby("class")
    a_df = grouped.get_group("A")
    b_df = grouped.get_group("B")
    e_df = grouped.get_group("E")
    v_df = grouped.get_group("V")

    total, a_list, b_list, e_list, v_list = [], [], [], [], []

    while len(a_list) + A < max(A, B, E, V):
        a_list.append(a_df.iloc[random.randrange(0, len(a_df))])

    while len(b_list) + B < max(A, B, E, V):
        b_list.append(b_df.iloc[random.randrange(0, len(b_df))])

    while len(e_list) + E < max(A, B, E, V):
        e_list.append(e_df.iloc[random.randrange(0, len(e_df))])

    while len(v_list) + V < max(A, B, E, V):
        v_list.append(v_df.iloc[random.randrange(0, len(v_df))])

    total = a_list+b_list+e_list+v_list

    return pd.concat([DataFrame, pd.DataFrame(total)])


DataFrame = random_over_sampling()
DataFrame = DataFrame.sample(frac=1)


print(train_validate_split())
print(cross_validation())
# testing()
