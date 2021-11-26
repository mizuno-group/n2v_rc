# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:12:53 2021

transform network structure to vectors (matrix) using node2vec

node2vec: Scalable Feature Learning for Networks.
Aditya Grover and Jure Leskovec.
Knowledge Discovery and Data Mining, 2016.

@author: I.Azuma
"""
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec

class Vectorization():
    def __init__(self):
        self.g = None
        self.obj_p = dict
        self.fit_p = dict
        self.vec_df = pd.DataFrame()
        self.model = None
    
    # set networkx.Graph class from bottom two ways
    def el2g(self,el_df):
        """
        set networkx Graph class from edge list dataframe
        
        Parameters
        ----------
        el_df : dataframe
        
            row    col   weight
            gene1  gene2   0.98
            gene3  gene4   0.85 
            gene5  gene6   0.62
        
        """
        if len(el_df) != 3:
            raise ValueError("ERROR! Set dataframe with its column size of 3.")
        else:
            pass
        el_df.columns = ['x','y','weight']
        el_df = el_df.sort_values('weight',ascending=False)
        S = el_df['x'].tolist()
        D = el_df['y'].tolist()
        W = el_df['weight'].tolist()
        
        G = nx.Graph()
        for i in range(len(S)):
            G.add_edge(S[i],D[i],weight=W[i])
        self.g = G
    
    def setg(self,G):
        """
        set networkx Graph class directly
        """
        self.g = G


    def set_model(self,model_path):
        """
        se pre-trained model
        # save model with 'model.save(model_path)'
        """
        self.model = Word2Vec.load(model_path)
    
    def set_parameter(self,dimensions=128,walk_length=100,num_walks=20,p=1,q=1,workers=4,window=15,min_count=1,sg=1):
        """
        parameter setting
        https://github.com/eliorc/node2vec
        1. dimensions : Embedding dimensions
        2. walk_length: Number of nodes in each walk
        3. num_walks: Number of walks per node
        4. p: Return hyper parameter
        5. q: Inout parameter
        6. workers: Number of workers for parallel execution
        """
        self.obj_p = {'dimensions':dimensions,'walk_length':walk_length, 'num_walks':num_walks, 'p':p, 'q':q, 'workers':workers}
        self.fit_p = {'window':window, 'min_count':min_count,'sg':sg}
        print("use these parameter")
        print(self.obj_p)
        print(self.fit_p)
        
    def conduct_vectorization(self):
        """
        this process may take a few minutes
        """
        n2v_obj = Node2Vec(self.g, dimensions=self.obj_p.get('dimensions'), walk_length=self.obj_p.get('walk_length'), num_walks=self.obj_p.get('num_walks'), p=self.obj_p.get('p'), q=self.obj_p.get('q'), workers=self.obj_p.get('workers'))
        self.model = n2v_obj.fit(window=self.fit_p.get('window'), min_count=self.fit_p.get('min_count'), sg=self.fit_p.get('sg'))
        
        norm_vec = self.model.wv.get_normed_vectors() # extract normalized vector
        names = self.model.wv.index_to_key
        self.vec_df = pd.DataFrame(norm_vec,index=names).astype(float) # vectors of node2vec
        
    def get_vec(self):
        return self.vec_df
        
    
