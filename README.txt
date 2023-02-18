Find more food reveiw data in folder find_more_data
	Get more data to balance and increase samples in 2 class and upload in to (link github)'https://github.com/cuongprotector/Food-Comments-Sentiment-Analysis'
include find_more_data.py,food_train.csv,more_food_cmt.csv,test.csv

How to run the model to generate submission results:
1. Model Bi_LSTM in folder Bi-LSTM
	Runing in google colaboratory, just run file Bi_LSTM.ipynb
		B1: install and import necessary library
		B2: prepare data to train and to submit
		B3: Visualize data and processing (VietNam tokenize, lemmatize)
		B4: Split train data for training and testing and encoding to sequence vector
		B5: Training model (Bi-LSTM), evaluate - pretrain , save model with file name Bi_LSTM.h5 in this folder
		B6: Predict with model in B4 with submission data and save file submission
Or can load model Bi_LSTM.h5 to predict test.csv in link github above
2. Model LSTM in folder LSTM
How to run the model to generate submission results:
	Runing in Kaggle, just run file LSTM.ipynb
		B1: install and import necessary library
		B2: prepare data to train and to submit
		B3: Visualize data and processing (VietNam tokenize, lemmatize)
		B4: Filter some special characters, Encoding data and convert to sequence vector 
		B5: Split train data for training and testing, training model (LSTM), evaluate - pretrain , save model with file name LSTM.h5 in this folder
		B6: Predict with model in B4 with submission data and save file submission
Or can load model LSTM.h5 to predict test.csv in link github above