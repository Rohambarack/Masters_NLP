The project is in the Proj/ folder

Proj/data -- data.csv is not uploaded can be reache from kaggle: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health

Proj/model_setup/encode_data.py -- script used for encoding data and saving it (it took a few hours (64 core CPU) 

Proj/model_setup/create_classifier.py -- script for creating the classifier from the encodings ( ~10 minnutes (16 core CPU))

Proj/src/classify.py -- small script to show how the classifier performs, enter a sentence and the classifier classifies it.

Proj/model/... plots and classification results. The classifier is also in there, so Proj/src/classify.py should run if the needed packages are installed
