# Regular imports
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

#NN classifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing



def main():
    #load classifier, encoder
    #with open('../model/encoder.pkl', "rb") as f:
    #    encoder_ml = pickle.load(f)
    encoder_ml = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    with open('../model/classifier.pkl', "rb") as f:
        classifier = pickle.load(f)

    labels = ['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Personality disorder', 'Stress', 'Suicidal']
    ### Take input
    print("Enter Text:, enter q  to quit")
    input1 = input()
    while input1 != "q":
        ### encode
        encoded_text = encoder_ml.encode(input1)
        encoded_text = encoded_text.reshape(1,-1)

        ### classify
        prediction = classifier.predict(encoded_text)
        ### reinterpret result
        
        prediction = labels[prediction[0]]

        print("Predicted label {}".format(prediction))
        print("Enter Text:, enter q  to quit")
        input1 = input()

if __name__ == "__main__":
    main()