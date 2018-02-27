import glob
import math
import codecs
import collections
from sklearn.cross_validation import train_test_split
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import gutenberg
from nltk.corpus import brown


def model_prob(X_train):
    vocab = set()
    n_unigram = collections.defaultdict(int)
    n_bigram = collections.defaultdict(int)

    for sentence in X_train:
        # word_list = word_tokenize(sentence)
        word_list = sentence
        word_list = ['*'] + word_list + ['STOP']
        for index, current_word in enumerate(word_list):
            if (index > 0 and index < len(word_list)):
                vocab.add(current_word)
                unigram_id = tuple([current_word])
                n_unigram[unigram_id] += 1

            if (index >= 0 and index < (len(word_list) - 1)):
                bigram_id = tuple([current_word, word_list[index + 1]])
                n_bigram[bigram_id] += 1

    # print len(n_unigram)
    # print len(n_bigram)

    t_word = 0
    for id in n_unigram:
        t_word += n_unigram[id]

    # print t_word

    p_unigram = collections.defaultdict(int)
    p_bigram = collections.defaultdict(int)

    for id in n_bigram:
        if (id[0] == '*'):
            temp = float(n_bigram[id]) / len(X_train)
            p_bigram[id] = math.log(temp, 2)
        else:
            prev_word = tuple([id[0]])
            temp = float(n_bigram[id]) / n_unigram[prev_word]
            p_bigram[id] = math.log(temp, 2)

    for id in n_unigram:
        temp = float(n_unigram[id]) / t_word
        p_unigram[id] = math.log(temp, 2)

    return (t_word, dict(p_unigram), dict(p_bigram))


def model_perplexity(total_words, uni_prob, bi_prob, X_test):
    length = 0
    prob = 0
    perplex = 0
    for sentence in X_test:
        # word_list = word_tokenize(sentence)
        word_list = sentence
        word_list = ['*'] + word_list + ['STOP']
        length += len(word_list)

        for index, current_word in enumerate(word_list):
            if (index < (len(word_list) - 1)):
                bigram_id = tuple([current_word, word_list[index + 1]])
                a = bi_prob.get(bigram_id, -100)
                if (a == -100):
                    unigram_id = tuple([current_word])
                    b = uni_prob.get(unigram_id, -100)
                    if (b == -100):
                        temp = 1 / float(total_words)
                        prob += math.log(temp, 2)
                    else:
                        prob += b
                else:
                    prob += a

    prob = prob / length
    perplex = 2 ** (-1 * prob)
    return perplex


'''data_path = '/home/aakash/Downloads/IISc/Sem2/NLU/Assignment1/Dataset/gutenberg/'
complete_file=glob.glob(data_path+'/*.txt')
corpus = list()

for f in complete_file:
    with codecs.open(f,'r',encoding='utf-8',errors='ignore') as file:
        df=file.read()
        f_sent = sent_tokenize(df)
        for s in f_sent:
            corpus.append(s)
    file.close()'''

data = list(gutenberg.sents(gutenberg.fileids()))
for i in range(len(data)):
    data[i] = list(map(lambda x: x.lower(), data[i]))

X_train_g, X_test_g = train_test_split(data, test_size=0.2)

total_words, uni_prob, bi_prob = model_prob(X_train_g)
perplexity = model_perplexity(total_words, uni_prob, bi_prob, X_test_g)

print ('Training at Gutenberg dataset and testing at gutenberg dataset then Perplexity is : ', perplexity)

'''data_path = '/home/aakash/Downloads/IISc/Sem2/NLU/Assignment1/Dataset/brown/'
complete_file=glob.glob(data_path+'/*')
corpus = list()
for f in complete_file:
    with codecs.open(f,'r',encoding='utf-8',errors='ignore') as file:
        df=file.read()
        f_sent = sent_tokenize(df)
        for s in f_sent:
            corpus.append(s)
    file.close()'''

data_b = list(brown.sents(brown.fileids()))
for i in range(len(data_b)):
    data_b[i] = list(map(lambda x: x.lower(), data_b[i]))

X_train_b, X_test_b = train_test_split(data_b, test_size=0.2)

total_words, uni_prob, bi_prob = model_prob(X_train_b)
perplexity = model_perplexity(total_words, uni_prob, bi_prob, X_test_b)

print ('Training at Brown dataset and testing at brown dataset then Perplexity is : ', perplexity)

X_train_combined = X_train_g+X_train_b

total_words,uni_prob,bi_prob = model_prob(X_train_combined)
perplexity = model_perplexity(total_words,uni_prob,bi_prob,X_test_g)
print ('Training at Combined dataset and testing at Gutenberg dataset then Perplexity is : ', perplexity)

perplexity = model_perplexity(total_words,uni_prob,bi_prob,X_test_b)
print ('Training at Combined dataset and testing at Brown dataset then Perplexity is : ', perplexity)