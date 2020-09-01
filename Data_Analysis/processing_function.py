
from pandas import np
from sklearn.feature_extraction.text import CountVectorizer

def put_sentece_together(sent):
    s = ' '.join(sent)
    return s

def return_top_common_words(filtered_sentences, top, n_gram=(1, 4)):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=n_gram)
    vec = vectorizer.fit(filtered_sentences)
    X = vec.transform(filtered_sentences)
    sum_words = X.sum(axis=0)
    words_freq = [[word, sum_words[0, idx]] for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    words = [w[0] for w in words_freq]
    return words[:top]


def sent_vectorizer(documents, model, fully_indexed):
    idx = 0
    result = np.zeros((len(documents), 300))
    for words in documents:
        list_of_numbers = []
        for w in words:
            if w in model:
                if w.lower() in fully_indexed[idx].keys():
                    list_of_numbers.append(fully_indexed[idx][w.lower()] * model[w])
                else:
                    list_of_numbers.append(model[w])
        result[idx] = np.mean(list_of_numbers or [np.zeros(model.vector_size)], axis=0)
        idx += 1
    return result