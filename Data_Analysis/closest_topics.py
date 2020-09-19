import itertools
import json

import networkx as nx
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_json

# import igraph


with open(r'.\Data\all.json') as f:
    data = json.load(f)
df = pd.DataFrame(data)

with open(r'.\Data_Analysis\most_frequent_tags.json', 'r') as f:
    n_most_frequent = json.load(f)

def heatmap(df, n_most_frequent, name):

    all_tags = set()
    for t in df['tags']:
        all_tags = all_tags.union(set(t))

    coocurences = pd.DataFrame(data=np.zeros((len(all_tags), len(all_tags))), index=all_tags, columns=all_tags,
                               dtype='int16')



    for t in df['tags']:

        list_of_nodes = list(itertools.combinations(t, 2))
        for n1, n2 in list_of_nodes:
            coocurences.at[n1, n2] += 1
            coocurences.at[n2, n1] += 1

    coocurences = coocurences.loc[n_most_frequent, n_most_frequent]


    sum_list = []
    for i in coocurences.index.to_list():
        sum = len(df.loc[df.apply(lambda x: i in x['tags'], axis=1)])
        sum_list.append(sum)
    coocurences['sum'] = sum_list
    first = coocurences.index.to_list() [0]
    last = coocurences.index.to_list()[-1]
    coocurences = coocurences.loc[:, first: last].div(coocurences["sum"], axis=0)


    sns.heatmap(coocurences)
    plt.savefig('./figures/'+name, bbox_inches='tight')
    plt.show()

# conn_indices = np.where(coocurences)
# # print(conn_indices)
# weights = coocurences.values
# edges = zip(*conn_indices)
# G = igraph.Graph(edges=edges, directed=False)
# G.vs['label'] = all_tags
# G.es['weight'] = weights
# G.es['width'] = weights
# igraph.plot(G, layout="rt", labels=True, margin=80)
