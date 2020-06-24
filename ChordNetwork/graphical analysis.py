
# coding: utf-8

# In[11]:


import numpy as np
import re
import requests
from bs4  import BeautifulSoup as soup
from tqdm import tqdm
import networkx as nx
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from pychord import Chord, ChordProgression
import itertools
import pickle
import os
import glob

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[38]:


import pandas as pd
import altair as alt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from pychord import Chord,ChordProgression
from synthesizer import Player, Synthesizer, Waveform
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from sklearn import datasets

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# In[3]:


def drawG(G, drawsize=10):
    weights = [0 for i in G.edges]
    for i,e in enumerate(G.edges):
        weights[i] =  np.log(e[2]+2)
    plt.figure(figsize=(drawsize,drawsize))
    nx.draw_networkx(G, width=weights )
    return G

class SongChords:
    def __init__(self, c, G):
        self.c = c
        self.G = G
    def get(self):
        return (self.c, self.G)


def to_weighted(G):
    M = G.copy()
    G = nx.DiGraph()
    for u,v,data in M.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if G.has_edge(u,v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)
    return G


def to_g(G):
    M = G.copy()
    G = nx.Graph()
    for u,v,data in M.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if G.has_edge(u,v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)
    return G

def getsong(num):
    full = f'./*/*{str(num)}.pickle'
    s = glob.glob(full)
    if len(s)>1: 
        print('more than one match??!!')
    else:
        print(s[0])
    s = pickle.load(open(s[0],'rb'))
    return s

def weighted_apl_clust(G, features = ['node','edge']):
#     return (nx.average_clustering(G),nx.average_shortest_path_length(G))
#     return (G.number_of_nodes(), G.number_of_edges())
#     return (nx.average_shortest_path_length(G), G.number_of_edges(), G.number_of_nodes())
    ret = {
        'node':0,
        'edge':0,
        'clust':0,
        'sdev-clust':0,
        'apl':0,
        'sdev-apl':0,
        'clust-g':0,
        'pagerank-max':0,
        'sdev-pagerank':0,
        'pagerank':0,
    }
    if 'apl' in features:
        ret['apl'] = nx.average_shortest_path_length(G)
    if 'sdev-apl' in features:
        n = []
        for i in [list(i[1].values())[1:] for i in nx.shortest_path_length(G)]:
            n+=i
        ret['sdev-apl'] = np.std(n)
    if 'sdev-apl-g' in features:
        n = []
        for i in [list(i[1].values())[1:] for i in nx.shortest_path_length(to_g(G))]:
            n+=i
        ret['sdev-apl-g'] = np.std(n)
    if 'apl-g' in features:
        ret['apl-g']  =  nx.average_shortest_path_length(to_g(G))
    if 'node' in features:
        ret['node'] = G.number_of_nodes()
    if 'edge' in features:
        ret['edge'] = G.number_of_edges()
    if 'clust' in features:
        ret['clust']=nx.average_clustering(to_weighted(G))
    if 'clust-g' in features:
        ret['clust-g']=nx.average_clustering(to_g(G))
    if 'sdev-clust' in features:
        ret['sdev-clust'] = np.std(list(nx.clustering(to_weighted(G)).values()))
    if 'sdev-clust-g' in features:
        ret['sdev-clust-g'] = np.std(list(nx.clustering(to_g(G)).values()))
    if 'pagerank-max' in features:
        ret['pagerank-max']=np.amax(list(nx.pagerank(to_weighted(G)).values()))
    if 'pagerank' in features:
        ret['pagerank']= np.average(list(nx.pagerank(to_weighted(G)).values()))
    if 'sdev-pagerank' in features:
        ret['sdev-pagerank'] = np.std(list(nx.pagerank(to_weighted(G)).values()))
    if 'sdev-pagerank-g' in features:
        ret['sdev-pagerank-g'] = np.std(list(nx.pagerank(to_g(G)).values()))
    return [ret[i] for i in features]


# In[54]:


names = [
    # 'pop',
    'jazz',
    'the_beatles_1916',
    'ramones_555', 
    'frank_sinatra_11333', 
    'metallica_954', 
    'linkin_park_1025',
    'passenger_21762',
    # 'bruno_mars_28280',
    # 'adele_20519',
    'elvis_presley_11125',
# #     'john_denver_10167',
    # 'blues',
    'etta_james_20433',
    'gary_moore_10959',
]


# features = ['node','apl', 'clust','sdev-apl','sdev-clust']
features = [
    'node',
    'edge',
    'clust',
    'sdev-clust',
    'apl',
    'sdev-apl',
    'clust-g',
    'pagerank-max',
    'sdev-pagerank',
    'pagerank',
]

songs = []
filelists = [f'./{n}-2d/*.edgelist' for n in names]
for fl in filelists:
    current = re.search(r'/([\w-]+)',fl)
    al = current.group(1)
    artist = glob.glob(fl)
    data = []
    for i in tqdm(artist):
        # create using ???
        G = nx.read_edgelist(i,comments='`',create_using=nx.MultiDiGraph)
        
        current = re.search(r'/([\w-]+).edgelist',i)
        name = current.group(1)
        # node_edge ???
        try:
            data.append([name, al]+list(weighted_apl_clust(G, features = features)))
        except:
            continue
    songs+=data

d = np.array(songs).T
data = {
    'name':d[0],
    'artist':d[1],
}
for n, f in enumerate(features):
    data[f]=d[n+2].astype(float)
df = pd.DataFrame(data)
df.head()


# In[55]:


musicians = []#pd.DataFrame([{'healp':0},{'healp':1}])
m_feats = []
for n in names:
    mus = {}
    mus['name'] = n
    n = n+'-2d'
    musician_data = df.loc[df['artist'] == n]
    for f in features:
        f_data = musician_data[f]
        mus[f+'-avg'] = np.average(f_data)
        mus[f+'-sdev'] = np.std(f_data)
    musicians.append(mus)
for f in features:
    m_feats+=[f+'-avg',f+'-sdev']
df_musician = pd.DataFrame(musicians)


# In[7]:


df_musician


# In[56]:


df0 = df.copy()
y = df0.pop('artist')
X = df0.drop('name', axis=1)
# X = df


# In[84]:


print(len(X), set(y))


# In[751]:


n_classes = np.unique(y).size

# Some noisy data not correlated
random = np.random.RandomState(seed=0)
E = random.normal(size=(len(X), 2200))

# Add noisy data to the informative features for make the task harder
X = np.c_[X, E]

svm = SVC(kernel='rbf', gamma='auto')
cv = StratifiedKFold(2)

print('beginnn')

score, permutation_scores, pvalue = permutation_test_score(
    svm, X, y, scoring="accuracy", cv=cv, n_permutations=10, n_jobs=-1, verbose=1)

print("Classification score %s (pvalue : %s)" % (score, pvalue))

# #############################################################################
# View histogram of permutation scores

plt.hist(permutation_scores, 20, label='Permutation scores',
         edgecolor='black')
ylim = plt.ylim()
# BUG: vlines(..., linestyle='--') fails on older versions of matplotlib
# plt.vlines(score, ylim[0], ylim[1], linestyle='--',
#          color='g', linewidth=3, label='Classification Score'
#          ' (pvalue %s)' % pvalue)
# plt.vlines(1.0 / n_classes, ylim[0], ylim[1], linestyle='--',
#          color='k', linewidth=3, label='Luck')
plt.plot(2 * [score], ylim, '--g', linewidth=3,
         label='Classification Score'
         ' (pvalue %s)' % pvalue)
plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')

print(2 * [score],np.average(permutation_scores),2 * [1. / n_classes])

plt.ylim(ylim)
plt.legend()
plt.xlabel('Score')
plt.show()
