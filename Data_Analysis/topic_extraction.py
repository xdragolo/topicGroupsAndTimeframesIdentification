# from gensim.models import Word2Vec
import nltk
import seaborn as sns
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize, pos_tag
import pandas as pd
from pandas import np
from sklearn.cluster import KMeans
import json
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from silhouette_tendency import silhouette_plot_tendency
from processing_function import sent_vectorizer, put_sentece_together, return_top_common_words

df = pd.read_json('../Data/covid.json')
documents = df['post_text']

### document tokenization to sentences and words
sentences = []
pos_sentences = []
for row in documents:
    tok_row = sent_tokenize(row)
    for sent in tok_row:
        sentences.append(word_tokenize(sent))
        pos_sentences.append(pos_tag(word_tokenize(sent), tagset='universal'))

### filtering stopwords
filtered_sentences = []
filtered_sentence = []
cnt_sentences = []
extra_stopwords = ["PS", "E.g", ")", "(", "!", '.', ',', "''", '``', "?", ":", "...", "the", "would", "one", "is", "if",
                   "is", "so", "should", "what", "am", "how", "sure", "really", "anything", "like", "even", "two",
                   "year", "the", "my", "also", "however", "if", "this", "could", "want", "without", "in", "so", "got",
                   "another", "still", "my", "much", "need", "able", "since", "ll", "get"]
wanted_pos = ['ADJ', 'VERB', 'NOUN', "ADV"]
for sentence in pos_sentences:
    filtered_sentence = [w[0] for w in sentence if
                         (w[0].lower() not in stopwords.words('english') + extra_stopwords and w[1] in wanted_pos)]
    cnt_sentences.append(' '.join(filtered_sentence))
    filtered_sentences.append(filtered_sentence)
    filtered_sentence = []


## use pretrained model
model = KeyedVectors.load_word2vec_format('./model/6/model.bin', binary=True)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cnt_sentences)
index_value = {i[1]: i[0] for i in vectorizer.vocabulary_.items()}
fully_indexed = []
for row in X:
    fully_indexed.append({index_value[column]: value for (column, value) in zip(row.indices, row.data)})
X = sent_vectorizer(filtered_sentences, model, fully_indexed)

# silhouette_plot_tendency(2,50,X)

# ### clustering word2vec vectors
NUM_CLUSTERS = 6
kmeans = KMeans(n_clusters=NUM_CLUSTERS, init='k-means++', max_iter=100, n_init=1)
kmeans.fit(X)
assigned_clusters = kmeans.predict(X)

### assign original sentences to clusters
clusters = {}

filtered_sentences = [put_sentece_together(s) for s in filtered_sentences]
filtered_sentences = np.array(filtered_sentences)

snt = [put_sentece_together(s) for s in sentences]
snt = np.array(snt, dtype=object)

distances = kmeans.transform(X)

for item in range(NUM_CLUSTERS):
    clusters[str(item)] = {}
    inx = np.where(assigned_clusters == item)

    clusters[str(item)]['common_words'] = return_top_common_words(filtered_sentences[inx], 6)
    d = distances[:, item]
    ind = np.argsort(d)[::][:5]

    clusters[str(item)]["closest_document"] = snt[ind].tolist()
    index = np.where(assigned_clusters == item)
    cl = snt[index]
    clusters[str(item)]["number_of_documents"] = len(cl)
    clusters[str(item)]['sentences'] = cl.tolist()

with open('covid_clusters.json', 'w', encoding="utf-8") as json_file:
    json.dump(clusters, json_file, indent=4, ensure_ascii=False)

pca = PCA(n_components=2).fit(X)
data2d = pca.transform(X)
centers2d = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers2d[:, 0], centers2d[:, 1],
            marker='+',
            color='black',
            s=200);

plt.scatter(data2d[:, 0], data2d[:, 1],
            c=assigned_clusters)
plt.savefig('./figures/cluster_visualisation.png')
plt.show()
