import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def draw_elbow(limit,X):
    sum_of_squared_distances = []
    K = range(1,limit)
    for true_k in K:
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=150, n_init=1,random_state=10)
        model.fit(X)
        sum_of_squared_distances.append(model.inertia_)

    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    plt.savefig('elbow.png')

df = pd.read_json('../Data/academia3.json')
documents = df['post_text']
vectorizer = TfidfVectorizer(ngram_range=(1,3), max_df=0.8, min_df=2,stop_words='english')
X = vectorizer.fit_transform(documents)
draw_elbow(100,X)