import json 
import pickle

import json 
import numpy as np
import random 
import re
import math
import pandas as pd
import pickle 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle 
from libact.models import LogisticRegression
from libact.models.svm import SVM
from libact.models import SklearnProbaAdapter
from libact.labelers import IdealLabeler
from libact.query_strategies import RandomSampling, UncertaintySampling
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models.sklearn_adapter import SklearnAdapter
from sklearn.neural_network import MLPClassifier

def dataset_preprocesing(X, y, n_labeled):
    
    X_train = list(X)[:int(len(X)*0.7)]
    y_train = list(y)[:int(len(y)*0.7)]
    
    # move first relevant document found to the first position, the same for X to start training 
    index_relevant = list(y).index(1)
    y_train.insert(0, list(y).pop(index_relevant))
    X_train.insert(0, list(X).pop(index_relevant))
        
    # create dataset instance with examples and labels 
    fully_labeled_ds = Dataset(X,y)
    
    # insert None values as unknown labels
    ds_unlabeled = Dataset(X_train, np.concatenate([y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    
    X_test = X[int(len(X)*0.7):]
    y_test = y[int(len(y)*0.7):]
    
    return X, y, X_test, y_test, ds_unlabeled, fully_labeled_ds

def Sort(sub_li): 
    sub_li.sort(key = lambda x: x[1], reverse=True) 
    return sub_li 

def return_model(MODEL):
     
    machine_learning_model = ''
    
    if MODEL == 'LR':
        machine_learning_model = LogisticRegression()

    elif MODEL == 'SVM_rbf':
        machine_learning_model = SklearnProbaAdapter(SVC(kernel='rbf',probability=True))
        
    elif MODEL == 'SVM_linear':
        machine_learning_model = SklearnProbaAdapter(SVC(kernel='linear',probability=True))
    
    elif MODEL == 'NB':
        machine_learning_model = SklearnProbaAdapter(BernoulliNB())

    elif MODEL == 'MLP':
        machine_learning_model = SklearnProbaAdapter(MLPClassifier())
        
    elif MODEL == 'RF':        
        machine_learning_model = SklearnProbaAdapter(RandomForestClassifier(n_estimators = 100))
    
    return machine_learning_model 


def return_embeddings_clef(representation, DATASET_DIR):
    
    dict_embeddings = {}
    embedding_dim = 0 
    
    if representation == 'TF-IDF':
        dict_embeddings = json.load(open('{}/clef_tfidf.json'.format(DATASET_DIR)))
        embedding_dim = 100
        
    elif representation == 'BERT':
        dict_embeddings = json.load(open('{}/clef_bert_embeddings.json'.format(DATASET_DIR)))
        embedding_dim = 768
    
    elif representation == 'BioBERT':
        dict_embeddings = json.load(open('{}/clef_bio_bert_embeddings.json'.format(DATASET_DIR)))
        embedding_dim = 768
    
    elif representation == 'W2VEC':
        dict_embeddings = json.load(open('{}/clef_w2vec.json'.format(DATASET_DIR)))
        embedding_dim = 300
    
    elif representation == 'GLOVE':
        dict_embeddings = json.load(open('{}/clef_glove_dict.json'.format(DATASET_DIR)))
        embedding_dim = 300
    
    return dict_embeddings, embedding_dim  


def return_embeddings_episte(representation, DATASET_DIR): 
    
    dict_embeddings = {}
    embedding_dim = 0 
    
    if representation == 'TF-IDF':
        dict_embeddings = json.load(open('{}/episte_tfidf.json'.format(DATASET_DIR)))
        embedding_dim = 100
        
    elif representation == 'BERT':
        dict_embeddings = json.load(open('{}/bert_embeddings.json'.format(DATASET_DIR)))
        embedding_dim = 768
    
    elif representation == 'BioBERT':
        dict_embeddings = json.load(open('{}/episte_bio_bert_embeddings.json'.format(DATASET_DIR)))
        embedding_dim = 768
    
    elif representation == 'W2VEC':
        dict_embeddings = json.load(open('{}/episte_w2vec.json'.format(DATASET_DIR)))
        embedding_dim = 300
    
    elif representation == 'GLOVE':
        dict_embeddings = json.load(open('{}/episte_glove_dict.json'.format(DATASET_DIR)))
        embedding_dim = 300
    
    return dict_embeddings, embedding_dim  
