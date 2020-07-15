import pickle
import json 


def dataset_preprocesing(X, y):
    
    X_train = list(X)[:int(len(X)*0.7)]
    y_train = list(y)[:int(len(y)*0.7)]
    
    X_test = list(X[int(len(X)*0.7):])
    y_test = list(y[int(len(y)*0.7):])
    
    index_relevant = list(y).index(1)
    y_test.insert(0, list(y).pop(index_relevant))
    X_test.insert(0, list(X).pop(index_relevant))
    
    return X, y, X_test, y_test


# input list of tuples and return sorted by second element (score)
def Sort(sub_li): 
    sub_li.sort(key = lambda x: x[1], reverse=True) 
    return sub_li 


def average_precision(lista):   
    precisions = []
    cum = 1 
    
    for i,x in enumerate(lista, start=1):
        if x ==1:
            precisions.append(cum/i)     
            cum +=1
    
    return sum(precisions)/cum

def last_rel(y):
    lastrel = ''.join([str(x) for x in y]).rindex('1')
    return lastrel/len(y)


def return_embeddings_episte(representation, DATASET_DIR): 
    
    dict_embeddings = {}
    
    if representation == 'TF-IDF':
        dict_embeddings = json.load(open('{}/episte_tfidf.json'.format(DATASET_DIR)))
        
    elif representation == 'BERT':
        dict_embeddings = json.load(open('{}/bert_embeddings.json'.format(DATASET_DIR)))
    
    elif representation == 'BioBERT':
        dict_embeddings = json.load(open('{}/episte_bio_bert_embeddings.json'.format(DATASET_DIR)))
    
    elif representation == 'W2VEC':
        dict_embeddings = json.load(open('{}/episte_w2vec.json'.format(DATASET_DIR)))
    
    elif representation == 'GLOVE':
        dict_embeddings = json.load(open('{}/episte_glove_dict.json'.format(DATASET_DIR)))
    
    return dict_embeddings 


def return_embeddings_clef(representation, DATASET_DIR):
    
    dict_embeddings = {}
    
    if representation == 'TF-IDF':
        dict_embeddings = json.load(open('{}/clef_tfidf.json'.format(DATASET_DIR)))
        
    elif representation == 'BERT':
        dict_embeddings = json.load(open('{}/clef_bert_embeddings.json'.format(DATASET_DIR)))
    
    elif representation == 'BioBERT':
        dict_embeddings = json.load(open('{}/clef_bio_bert_embeddings.json'.format(DATASET_DIR)))
    
    elif representation == 'W2VEC':
        dict_embeddings = json.load(open('{}/clef_w2vec.json'.format(DATASET_DIR)))
    
    elif representation == 'GLOVE':
        dict_embeddings = json.load(open('{}/clef_glove_dict.json'.format(DATASET_DIR)))
    
    return dict_embeddings 
    

def machine_learning_model_clef(chosen_model, question, representation, DATASET_DIR): 
    
    ml_model = ''
    
    if representation == 'TF-IDF':
        ml_model = pickle.load(open('{}/models_clef/tf_idf/{}/{}_tfidf_clef.sav'.format(DATASET_DIR, chosen_model, question),'rb'))  
        
    elif representation == 'BERT':
        ml_model = pickle.load(open('{}/models_clef/BERT/{}/{}_BERT_clef.sav'.format(DATASET_DIR,chosen_model, question),'rb'))
    
    elif representation == 'BioBERT':
        ml_model = pickle.load(open('{}/models_clef/BioBERT/{}/{}_BioBERT_clef.sav'.format(DATASET_DIR,chosen_model, question),'rb'))
    
    elif representation == 'W2VEC':
        ml_model = pickle.load(open('{}/models_clef/w2vec/{}/{}_w2vec_clef.sav'.format(DATASET_DIR,chosen_model, question),'rb'))
    
    elif representation == 'GLOVE':
        ml_model = pickle.load(open('{}/models_clef/glove/{}/{}_glove_clef.sav'.format(DATASET_DIR,chosen_model, question),'rb'))
    
    return ml_model 


def machine_learning_model_episte(chosen_model, question, representation, DATASET_DIR): 
    
    ml_model = ''
    
    if representation == 'TF-IDF':
        ml_model = pickle.load(open('{}/models_episte/tf_idf/{}/{}_tfidf_episte.sav'.format(DATASET_DIR, chosen_model, question),'rb'))  
        
    elif representation == 'BERT':
        ml_model = pickle.load(open('{}/models_episte/BERT/{}/{}_BERT_episte.sav'.format(DATASET_DIR, chosen_model, question),'rb'))
    
    elif representation == 'BioBERT':
        ml_model = pickle.load(open('{}/models_episte/BioBERT/{}/{}_BioBERT_episte.sav'.format(DATASET_DIR, chosen_model, question),'rb'))
    
    elif representation == 'W2VEC':
        ml_model = pickle.load(open('{}/models_episte/w2vec/{}/{}_w2vec_episte.sav'.format(DATASET_DIR, chosen_model, question),'rb'))
    
    elif representation == 'GLOVE':
        ml_model = pickle.load(open('{}/models_episte/glove/{}/{}_glove_episte.sav'.format(DATASET_DIR, chosen_model, question),'rb'))
    
    return ml_model 




    