# from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from nltk import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans
import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances_argmin_min

df = pd.read_json('../Data/covid.json')
documents = df['post_text']

def put_sentece_together(sent):
    s = ' '.join(sent)
    return s

def return_top_common_words(filtered_sentences, top):
    vectorizer = CountVectorizer(analyzer='word',ngram_range=(1,4))
    vec = vectorizer.fit(filtered_sentences)
    X = vec.transform(filtered_sentences)
    sum_words = X.sum(axis=0)
    words_freq = [[word, sum_words[0, idx]] for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    words = [w[0] for w in words_freq]
    return words[:top]


### document tokenization to sentences and words
sentences = []
for row in documents:
    tok_row = sent_tokenize(row)
    for sent in tok_row:
        sentences.append(word_tokenize(sent))



### filtering stopwords
filtered_sentences = []
filtered_sentence = []
extra_stopwords = ["PS","E.g",")","(","!",'.',',',"''",'``',"?",":","...","the","would","one","is","if","is","so","should","what","am","how","sure","really","anything","like","even","two","year"]
for sentence in sentences:
    for w in sentence:
        if w not in stopwords.words('english') and w not in extra_stopwords:
            filtered_sentence.append(w)
    filtered_sentences.append(filtered_sentence)
    filtered_sentence = []

### train own model
# model = Word2Vec(filtered_sentences, min_count=1)

### or use pretrained model
model = KeyedVectors.load_word2vec_format('./model/6/model.bin', binary=True)


def sent_vectorizer(documents, model):
    return np.array([
        np.mean([model[w] for w in words if w in model] or
                [np.zeros(model.vector_size)], axis=0)
        for words in documents])


X = sent_vectorizer(filtered_sentences, model)
# for words in sentences[0]:
#     if words in model:
#         print(model[words])

### clustering word2vec vectors
NUM_CLUSTERS = 15
kmeans = KMeans(n_clusters=NUM_CLUSTERS, init='k-means++', max_iter=100, n_init=1)
assigned_clusters = kmeans.fit_predict(X)

# s
# f = open('report.txt','w')
# for i in range(15):
#     d = kmeans.transform(X)[:,i]
#     print(d)
#     ind = np.argsort(d)[::][:5]
#     print(i, file=f)
#     print(np.array(sentences, dtype=object)[ind],file=f)
#     print('_______________________________________________', file=f)

### assign original sentences to clusters
clusters = {}
n = 0
filtered_sentences = [put_sentece_together(s) for s in filtered_sentences]
filtered_sentences = np.array(filtered_sentences)
snt = [put_sentece_together(s)for s in sentences[:]]
snt = np.array(snt, dtype=object)

for item in range(NUM_CLUSTERS):
    clusters[str(item)] = {}
    cl = put_sentece_together(sentences[n])
    inx = np.where(assigned_clusters == item)
    clusters[str(item)]['common_words'] = return_top_common_words(filtered_sentences[inx], 6)
    d = kmeans.transform(X)[:, item]
    ind = np.argsort(d)[::][:5]
    clusters[str(item)]["closest_document"] = snt[ind].tolist()
    index = np.where(assigned_clusters == item)
    cl = snt[index]
    clusters[str(item)]["number_of_documents"] = len(cl)
    clusters[str(item)]['sentences'] = cl.tolist()


# for item in assigned_clusters:
#     cl = {}
#
#     if str(item) in clusters.keys():
#         cl = put_sentece_together(sentences[n])
#         clusters[str(item)]['sentences'].append(cl)
#     else:
#
#
#
#     n += 1


with open('covid_clusters.json','w',encoding="utf-8") as json_file:
    json.dump(clusters,json_file, indent=4,ensure_ascii=False)




