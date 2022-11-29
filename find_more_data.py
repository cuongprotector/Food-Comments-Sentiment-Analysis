# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 02:33:19 2022

@author: nygeu
"""
import pandas as pd
import numpy as np
import os
# read file train csv
food_cmt_df = pd.read_csv("C:/Users/nygeu/OneDrive/Máy tính/Hocmay/full_train.csv/full_train.csv",encoding="UTF8")
food_cmt_df.head(10)

food_train = food_cmt_df[["Comment","Rating"]].copy()
# check vl
check_NaN = food_train.isna()
for i in range(len(food_train["Rating"])):
    if food_train["Rating"][i] != 1 and food_train["Rating"][i] != 0:
        food_train.drop(index=i,inplace=True)
print(food_train["Rating"].value_counts())
#   with 7146 1 and 1924 0 this label data will lost balance so we will add more data with label 0,1 until this samples reach 10k for each 0,1
# We get more data for food reviewer content in website "https://github.com/congnghia0609/ntc-scv/tree/master/data"
# I get 30k sample Equally divided to positive and negative
data_train_path_0 = "C:/Users/nygeu/OneDrive/Máy tính/Hocmay/data_train/train/neg"
data_train_path_1 = "C:/Users/nygeu/OneDrive/Máy tính/Hocmay/data_train/train/pos"

data_train_list_0 = os.listdir(data_train_path_0)
data_train_list_1 = os.listdir(data_train_path_1)

content_0 = []
content_1 = []

for i in range(8076):
    f = open(data_train_path_0+"/"+data_train_list_0[i],mode='r',encoding=("UTF8"))
    text_vl = f.read()
    content_0.append([text_vl,0])

for i in range(2854):
    f = open(data_train_path_1+"/"+data_train_list_1[i],mode='r',encoding=("UTF8"))
    text_vl = f.read()
    content_1.append([text_vl,1])
del f,i,text_vl
content = content_0+content_1
c_matrix = np.array(content)
c_matrix.shape
d = {'Comment':c_matrix[:,0].reshape(len(c_matrix)),'Rating':c_matrix[:,1].reshape(len(c_matrix))}
more_food_cmt = pd.DataFrame(data=d)
more_food_cmt = more_food_cmt.sample(frac = 1).reset_index(drop=True)
more_food_cmt.to_csv("more_food_cmt.csv")
# concat 2 DataFrame with 10k samples 1 and 10k samples 0
food_train = pd.concat([food_train,more_food_cmt])
# remove  value duplication and keep only one 
H = pd.DataFrame(data = food_train.drop_duplicates())
H.Rating = H.Rating.astype('int64')
print(H["Rating"].value_counts())
H.to_csv("food_train.csv")
