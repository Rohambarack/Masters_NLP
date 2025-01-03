#install packs
import os
os.system("pip install -U sentence-transformers")
os.system("pip install numpy")
os.system("pip install scikit-learn")
os.system("pip install pandas")
os.system("pip install turftopic")
os.system("pip install plotly")
os.system("pip install tensorflow")
os.system("pip install tf-keras")
os.system("pip install tensorflow_hub")
os.system("pip install bert")
os.system("pip install matplotlib")
#load packs
# Regular imports
import numpy as np
import pandas as pd
import tqdm # for progress bar
import math
import random
import re
import pickle


import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Tensorflow Import
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

# Bert Import for Tokenizer
import bert

#NN classifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
##################################### functions
# function to get unique values
def unique(list1):
    unique_list = pd.Series(list1).drop_duplicates().tolist()
    return unique_list

# function to flatten a list of lists
def flatten(xss):
    return [x for xs in xss for x in xs]

#make mini test set 200 from each tag
def separate_mini_indices(df_tag_column, nr = 200):
    #get tag list of unique tags
    list_of_tags = unique(df_tag_column)

    indices_list = []
    for tag in list_of_tags:
        #get indices of all for each tag
        tag_indices = [i for i, x in enumerate(df_tag_column) if x == tag]
        #shorten
        tag_short_indices = tag_indices[0:nr]
        #append to all
        indices_list.append(tag_short_indices)

        #makeshift progress info for bugfixing
        all_t = len(list_of_tags)
        current = list_of_tags.index(tag) + 1
        print("Tag: {} indexed. {} / {}".format(tag, current, all_t))
    
    flat_list = flatten(indices_list)

    return flat_list

def make_mini_df(df,df_tag_column, nr = 200):
    #get tag indices
    tag_indeces = separate_mini_indices(df_tag_column, nr)
    #subset df
    df_mini_list = [] 
    for i in tag_indeces:

        df_mini = df.iloc[[i]]
        df_mini_list.append(df_mini)
    
    df_2 = pd.concat(df_mini_list)

    return df_2

#################################### main
def main():
    #data
    df = pd.read_csv("../data/data.csv")
    #encoder
    encoder_ml = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    ################clean
    #clean data from NaNs
    check_list = []
    for i in df["statement"]:
        t = type(i)
        check_list.append(t)
        
    #find all droppable indices
    drop_indices = [i for i, x in enumerate(check_list) if x == float]

    #remove them
    df_c = df.drop(drop_indices)
    ###################################
    print("encoding starts:....")
    #encode all statements
    encoded_list = []
    length_full = len(df_c["statement"])
    current_count = 1
    milestones = np.arange(1, length_full, 1000).tolist()

    for statement in df_c["statement"]:
        encoded = encoder_ml.encode(statement)
        encoded_list.append(encoded)
        #makeshift progress track
        progress_b = round(current_count/length_full*100,2)
        if current_count in milestones:
            print("Progress at {} %".format(progress_b))
        else:
            pass
        current_count += 1

    #save output
    #save encoded  list
    filename = '../model/encoded_l_all.pkl'
    pickle.dump(encoded_list, open(filename, 'wb')) 

if __name__ == "__main__":
    main()