

# importing related modules

from node2vec import Node2Vec
import networkx as nx

#importing  adjacency list file as B
B = nx.read_adjlist("my.adjlist")

seed_value=0


# Specifying the input and hyperparameters of the node2vec model
node2vec = Node2Vec(B, dimensions=32, walk_length=15, num_walks=100, workers=1,seed=seed_value,temp_folder = './tmp')  

#Assigning/specifying random seeds

import os
os.environ['PYTHONHASHSEED']=str(seed_value)

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)


# creation of the model

model = node2vec.fit(window=10, min_count=1, batch_words=4,seed=seed_value)   


# saving the output vector

model.wv.save_word2vec_format("vectors.emb")

# save the model
model.save("vectorMODEL")

