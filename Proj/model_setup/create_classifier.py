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
import itertools

import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Tensorflow Import
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
#plots
import matplotlib.pyplot as plt

#NN classifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

#######################################

def main():
    #read data
    df = pd.read_csv("../data/data.csv")
    #clean data
    #clean data fro NaNs
    check_list = []
    for i in df["statement"]:
        t = type(i)
        check_list.append(t)
        
    #find all droppable indices
    drop_indices = [i for i, x in enumerate(check_list) if x == float]

    #remove them
    df_c = df.drop(drop_indices)

    #lets see
    with open('../model/encoded_l_all.pkl', "rb") as f:
        e_list = pickle.load(f)

    #reindex df and add encodings (for bugfixing reasons)
    new_index = list(itertools.chain(range(len(df_c["status"]))))
    df_c = df_c[["statement","status"]].set_axis(new_index,axis = "index")
    df_c["encoded"] = e_list

    ########################the star of the show :)
    classifier  = MLPClassifier(
                    #sizes based on trial
                    hidden_layer_sizes = (1000,),
                    #relu instead of logistic
                    activation = "relu",
                    #performs good according to doc, no need to try with different learning rates
                    solver = "adam",
                    #smaller alpha, less underfitting in docs def = 0.0001
                    alpha = 0.01,
                    batch_size = 200, #auto is 200 seems reasonable? maybe more for the large?
                    learning_rate_init = 0.0005, # less than default, more epochs but less fluct
                    #IMPORTANT beta and epsilon terms of "adam" are left at default
                    #large epoch number, we'l set to so probably won't reach anyway
                    max_iter=200,
                    #shuffle and mini batches help with leaving a local minima
                    shuffle = True,
                    #setting tolerance and n_iter no change and validation to stop early
                    tol = 0.001, #default is  smaller, but it's good for us
                    n_iter_no_change = 5, #smaller than default
                    #improvement in loss under 5 epochs, stops
                    early_stopping = True,
                    validation_fraction = 0.1, #10 percent is cool
                    verbose = True, # I like outputs :)
                    random_state = 42)
    
    ##########################split train and test
    #create X and y
    X_embeds = df_c["encoded"].values
    y_labs = df_c["status"].values

    X_train_l, X_test_l, y_train, y_test = train_test_split(
        X_embeds, y_labs, test_size=0.2, random_state=42, stratify =y_labs)
    #refine X and y to correct form

    #label encoder
    le = preprocessing.LabelEncoder()
    new_labels_test = le.fit_transform(y_test)
    new_labels_train = le.fit_transform(y_train)
    #create large 0 filled and fill row by row (according to forums thats the "efficient" way)
    X_train = np.zeros(shape=(len(X_train_l),768))
    X_test = np.zeros(shape=(len(X_test_l),768))
    #get embeddings to correct shape
    for i in range(len(X_train_l)):
        X_train[i] = X_train_l[i]

    for i in range(len(X_test_l)):
        X_test[i] = X_test_l[i]
        
    ####################################fit model
    clf = classifier.fit(X_train,new_labels_train)

    ########################Evaluate
    # loss curve
    plt.plot(classifier.loss_curve_)
    plt.title("Loss curve during training", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Loss score')
    plt.savefig('../model/loss_curve.png')

    #clear plot
    plt.clf()

    #validatin
    plt.plot(classifier.validation_scores_)
    plt.title("Validation scores during training", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.savefig('../model/valid_curve.png')

    ###################################### evaluate test
    prediction = classifier.predict(X_test)
    reinverted_labels = le.inverse_transform([0,1,2,3,4,5,6])
    cr = metrics.classification_report(new_labels_test, prediction, target_names=reinverted_labels)

    #save output
    f = open('../model/NN_report_1000.txt', 'w')
    f.write('Neural Network Classifier output\n\nClassification Report\n\n{}'.format(cr))
    f.close()

    #save classifier
    # save the model 
    filename = '../model/classifier.pkl'
    pickle.dump(classifier, open(filename, 'wb')) 


if __name__ == "__main__":
    main()