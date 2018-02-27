import glob
import math
import codecs
import collections
from sklearn.cross_validation import train_test_split
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import gutenberg
from nltk.corpus import brown


def preprocessing(lamda, X_train):
    vocab = set()
    n_unigram = collections.defaultdict(int)
    n_bigram = collections.defaultdict(int)
    next_word = collections.defaultdict(set)
    alpha = collections.defaultdict(set)

    for sentence in X_train:
        # word_list = word_tokenize(sentence)
        word_list = sentence
        word_list = ['*'] + word_list + ['STOP']
        for index, current_word in enumerate(word_list):
            if (index > 0 and index < len(word_list)):
                vocab.add(current_word)
                unigram_id = tuple([current_word])
                n_unigram[unigram_id] += 1
                '''
                if unigram_id not in n_unigram:
                    n_unigram[unigram_id]=1
                else:
                    n_unigram[unigram_id] += 1
                 '''

            if (index >= 0 and index < (len(word_list) - 1)):
                bigram_id = tuple([current_word, word_list[index + 1]])
                unigram_id = tuple([current_word])
                n_bigram[bigram_id] += 1
                next_word[unigram_id].add(word_list[index + 1])
                '''
                if bigram_id not in n_bigram:
                    n_bigram[bigram_id] = 1
                    next_word_id = tuple([current_word])
                    next_word[next_word_id].add[word_list[index+1]]
                else:
                    n_bigram[bigram_id] +=1
                '''

    final_count = n_bigram

    for bigram in n_bigram:
        final_count[bigram] = n_bigram[bigram] - lamda

    for current_word in vocab:
        unigram_id = tuple([current_word])
        alpha[unigram_id] = (len(next_word[unigram_id]) * lamda) / n_unigram[unigram_id]

    # print (len(final_count))
    # print (len(alpha))

    return (vocab, n_unigram, final_count, next_word, alpha)


def model_perplexity(X_test, vocab, n_unigram, final_count, next_word, alpha):
    prob = 0
    n_total_unigram = 0
    # n_next_word = 0
    length = 0
    for word in vocab:
        n_total_unigram += n_unigram[tuple([word])]

    # for word in n_unigram:
    #    n_next_word += n_unigram[tuple([word])]

    for sentence in X_test:
        # word_list = word_tokenize(sentence)
        word_list = sentence
        word_list = ['*'] + word_list + ['STOP']
        length += len(word_list)

        for index, current_word in enumerate(word_list):
            if (index > 0 and index < (len(word_list) - 1)):
                unigram_id1 = tuple([word_list[index + 1]])
                unigram_id2 = tuple([current_word])
                bigram_id = tuple([current_word, word_list[index + 1]])

                if bigram_id in final_count:
                    if (n_unigram[tuple([current_word])] == 0):
                        print(current_word)
                    temp = (final_count[bigram_id]) / (n_unigram[tuple([current_word])])
                    if (temp <= 0):
                        print(temp)
                        print(final_count[bigram_id])
                        print(n_unigram[tuple([current_word])])

                    prob += math.log(temp, 2)
                else:
                    if ((unigram_id1 in n_unigram) and (unigram_id2 in n_unigram)):
                        n_next_word = 0
                        for word in next_word[tuple([current_word])]:
                            n_next_word += n_unigram[tuple([word])]
                        # print (n_unigram[tuple([word_list[index+1]])])
                        # print (n_total_unigram)
                        # print (n_next_word)
                        temp = n_unigram[tuple([word_list[index + 1]])] / (n_total_unigram - n_next_word)
                        # print ('temp: ',temp)
                        temp = temp * alpha[tuple([current_word])]
                        # print ('temp: ',temp)
                        prob += math.log(temp, 2)
                    else:
                        temp = 1 / n_total_unigram
                        prob += math.log(float(temp), 2)

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

X_train_g, remaining = train_test_split(data, test_size=0.2)
X_dev_g, X_test_g = train_test_split(remaining, test_size=0.5)

min_perplexity = 1000000
for i in range(1, 10):
    l = i * 0.1
    vocab, n_unigram, final_count, next_word, alpha = preprocessing(l, X_train_g)
    perplexity = model_perplexity(X_dev_g, vocab, n_unigram, final_count, next_word, alpha)
    if (perplexity < min_perplexity):
        min_perplexity = perplexity
        lamda = l

# lamda = 0.1
vocab, n_unigram, final_count, next_word, alpha = preprocessing(lamda, X_train_g + X_dev_g)
perplexity = model_perplexity(X_test_g, vocab, n_unigram, final_count, next_word, alpha)
print ('Training at Gutenberg dataset and testing at gutenberg dataset then Perplexity is : ', perplexity)
# print len(vocab)


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

X_train_b, remaining = train_test_split(data_b, test_size=0.2)
X_dev_b, X_test_b = train_test_split(remaining, test_size=0.5)

min_perplexity = 1000000
for i in range(1, 10):
    l = i * 0.1
    vocab, n_unigram, final_count, next_word, alpha = preprocessing(l, X_train_b)
    perplexity = model_perplexity(X_dev_b, vocab, n_unigram, final_count, next_word, alpha)
    if (perplexity < min_perplexity):
        min_perplexity = perplexity
        lamda = l

vocab, n_unigram, final_count, next_word, alpha = preprocessing(lamda, X_train_b + X_dev_b)
perplexity = model_perplexity(X_test_b, vocab, n_unigram, final_count, next_word, alpha)
print ('Training at Brown dataset and testing at brown dataset then Perplexity is : ', perplexity)


X_train_combined = X_train_g+X_train_b
X_dev_combined = X_dev_g+X_dev_b

min_perplexity = 1000000
for i in range(1,10):
    l=i*0.1
    vocab, n_unigram, final_count, next_word, alpha = preprocessing(l,X_train_combined)
    perplexity = model_perplexity(X_dev_combined, vocab, n_unigram, final_count, next_word, alpha)
    if (perplexity<min_perplexity):
        min_perplexity = perplexity
        lamda = l

vocab, n_unigram, final_count, next_word, alpha = preprocessing(lamda,X_train_combined+X_dev_combined)

perplexity = model_perplexity(X_test_g, vocab, n_unigram, final_count, next_word, alpha )
print ('Training at Combined dataset and testing at Gutenberg dataset then Perplexity is : ', perplexity)

perplexity = model_perplexity(X_test_b, vocab, n_unigram, final_count, next_word, alpha )
print ('Training at Combined dataset and testing at Brown dataset then Perplexity is : ', perplexity)