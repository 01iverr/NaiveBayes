import tensorflow as tf
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
from collections import Counter
import matplotlib.pyplot as plt
import math

skip = 50
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=200, skip_top=skip)

word_index = tf.keras.datasets.imdb.get_word_index()
index2word = dict((i + 3, word) for (word, i) in word_index.items())
index2word[0] = '[pad]'
index2word[1] = '[bos]'
index2word[2] = '[oov]'

x_train = np.array([' '.join([index2word[idx] for idx in text]) for text in x_train])
x_test = np.array([' '.join([index2word[idx] for idx in text]) for text in x_test])

# create vocabulary
vocabulary = list()
for text in x_train:
    tokens = text.split()
    vocabulary.extend(tokens)

vocabulary = set(vocabulary)

# create binary vectors
x_train_binary = list()
x_test_binary = list()

for text in tqdm(x_train):
    tokens = text.split()
    binary_vector = list()
    for vocab_token in vocabulary:
        if vocab_token in tokens:
            binary_vector.append(1)
        else:
            binary_vector.append(0)
    x_train_binary.append(binary_vector)

x_train_binary = np.array(x_train_binary)

for text in tqdm(x_test):
    tokens = text.split()
    binary_vector = list()
    for vocab_token in vocabulary:
        if vocab_token in tokens:
            binary_vector.append(1)
        else:
            binary_vector.append(0)
    x_test_binary.append(binary_vector)

x_test_binary = np.array(x_test_binary)


def get_entropy(data):
    """
    # data: lista dyadikou dianysmatos
    """
    c = Counter(data)  # lexiko me to plhthos twn 0 kai 1
    if c[0] == 0 or c[1] == 0:
        return 0  # ama ola ta stoixeia einai se mia kathgoria eimai apolyta bebaios
    return -(c[0] / len(data)) * math.log2(c[0] / len(data)) - (c[1] / len(data)) * math.log2(c[1] / len(data))


def attr_entropy(x_data, y_data, label):
    """
    # x_data: lista me listes dyadikwn dianysmatwn (keimeno kritikhs)
    # y_data: lista dyadikou dianysmatos (thetikh/arnhtikh kritikh)
    # label: string lexi apo tis kritikes
    """
    if label not in word_index:
        return
    col = word_index[label]-skip  # se poia thesi brisketai h sygkekrimenh lexi
    has_counter_vector = []  # dyadiko dianisma opoy gia kathe kritikh opou yparxei h leksi se aythn kai me 1 einai thetikh kai 0 arnhtikh
    not_has_counter_vector = []  # dyadiko dianisma opoy gia kathe kritikh opou  DEN yparxei h lexi se aythn kai me 1 einai thetikh kai 0 arnhtikh
    for i in range(len(x_data)): #diatrexei to mege8os twn kritikwn
        if x_data[i][col]: #an periexetai h leksh sthn ekastote kritikh
            if y_data[i]: 
                has_counter_vector.append(1)  # thetikh prosthetei 1
            else:
                has_counter_vector.append(0)  # arnhtikh prosthetei 0
        else:#an dnperiexetai h leksh sthn ekastote kritikh
            if y_data[i]:
                not_has_counter_vector.append(1)  # thetikh prosthetei 1
            else:
                not_has_counter_vector.append(0)  # arnhtikh prosthetei 0
    has_at_en = 0 
    if len(has_counter_vector) > 0: #exei emfanistei h leksh se rkitikh
        has_at_en = get_entropy(has_counter_vector)

    not_has_at_en = 0
    if len(not_has_counter_vector) > 0: #an dn exei emfanistei h leksh se kamia kritikh 
        not_has_at_en = get_entropy(not_has_counter_vector)

    return (len(has_counter_vector) / len(x_data)) * has_at_en + (
                len(not_has_counter_vector) / len(x_data)) * not_has_at_en #typos entropias gia thn sygkekrimenh leksh



def info_gain(x_data, y_data, label):
    """
    # x_data: lista me listes dyadikwn dianysmatwn (keimeno kritikhs)
    # y_data: lista dyadikou dianysmatos (thetikh/arnhtikh kritikh)
    # label: string lexi apo tis kritikes
    """
    return get_entropy(y_data) - attr_entropy(x_data, y_data, label) #typos gia to poso xrhsimh einai mia leksh


def top_attributes(x_data, y_data, top_words):
    top_ig = []
    for i in vocabulary:
        if i in ['[pad]', '[bos]', '[oov]']:
            continue
        q = info_gain(x_data, y_data, i)
        top_ig.append((i, q))
    top_ig.sort(key=lambda x: x[1])
    return top_ig[:top_words] #epistrefei tis top lekseis me bash to info gain sotarismenes


class BernoulliNaiveBayes:
    def __init__(self, laplace = 1): #an dn yparxei kapoia leksh na mhn dhmiourgh8ei problhma me ton typo kaleitai sthn 150
        self.laplace = laplace
        return
    
    def fit(self, X, y):
        y_counter = Counter(y) # how many times each appears (number of 0s and 1s)
        
        # P(y) 
        class_prior =  [y_counter[0] / (y_counter[0]+y_counter[1]), 
                        y_counter[1] / (y_counter[0]+y_counter[1])]
        self.class_prior = np.expand_dims(class_prior, axis = 1)
        
        # P(x|y)
        likelyhood = np.zeros([2, len(X[0])]) # 2 because 2 binary values (0,1)
        
        for i in range(2): # 2 because 2 binary values (0,1)
            # only get rows from y
            y_rows = []
            for j in y:
                y_rows.append(j == i) 
            y_rows = np.array(y_rows)
            
            # P(x|y) = P(x and y) / P(y)
            likelyhood[i] = (X[y_rows].sum(axis = 0) + self.laplace) / (X[y_rows].shape[0] + 2 * self.laplace) #axis = 0 cause: sum of columns in X
        
        # probabilities for P(x|y) and P(not x|y)
        self.likelyhood_pos = likelyhood 
        self.likelyhood_neg = 1 - likelyhood 
            
    def predict(self, X):
        # [0, 1, 0, 1, 0]
        # [1, 1, 0, 1, 0]
        # [0, 0, 0, 1, 0]
        #
        # [0,1,0]
        # [1,1,0]
        # [0,0,0]
        # [1,1,1]
        # [0,0,0]
        
        # polaplasiasmos me ton anastrofo wste kathe grammh na anaferetai sthn antistoixh lexh
        probs_pos = self.likelyhood_pos.dot(X.T) 
        probs_neg = self.likelyhood_neg.dot(1 - X.T) # antistoixa gia ta arnhtika
        likelyhoods = probs_pos + probs_neg

        # self.class_prior: poses thetikes kai arnhtikes kritikes yparxoyn sto synolo twn dedomenwn ekfrazmena ws pithanothtes
        joint_likelyhoods = likelyhoods + self.class_prior 

        # get y that maximizes P(y|x)
        preds = np.argmax(joint_likelyhoods, axis = 0) # me to axis=0 elegxei se poia sthlh einai h megalyterh pithanothta kai epistrefei th seira poy brhke th megalyterh opou einai h antistoixh kritikh
        return preds


voc = top_attributes(x_train_binary, y_train, 100)
vocabulary = set()
for i in voc:
    vocabulary.add(i[0])

bnb = BernoulliNaiveBayes(laplace = 1)
bnb.fit(x_train_binary, y_train)
prediction = bnb.predict(x_test_binary)
print(classification_report(y_test, prediction))

# by sklearn for comparison
print("\nBy sklearn:")
from sklearn.naive_bayes import BernoulliNB
nb = BernoulliNB()
nb.fit(x_train_binary, y_train)
print(classification_report(y_test, nb.predict(x_test_binary)))

