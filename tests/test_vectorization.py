# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 10:54:46 2021

@author: I.Azuma
"""
import unittest
import networkx as nx
import pandas as pd
import os
import sys

from node2vec_recursive_clustering.vectorizationn import Vectorization

BASEPATH = os.path.dirname(os.path.abspath(__file__))

class SampleTest(unittest.TestCase):
    SETUP = None
    
    # call when test class initialization
    @classmethod
    def setUpClass(cls):
        print('****** setUpClass method is called. ******')
        cls.SETUP = nx.read_gpickle(BASEPATH.replace("tests","tests/sample_data/sample_cor.gpickle"),index_col=0)
    
    # called when test class end
    @classmethod
    def tearDownClass(cls):
        print('****** setDownClass method is called. ******')
    
    # called when a test method runs
    def setUp(self):
        self.smpl = Vectorization() # initialization must be done
        self.smpl.setg(SampleTest.SETUP)

    # called when a test method ends
    def tearDown(self):
        pass
    
        