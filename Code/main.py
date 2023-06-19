# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import preprocess
import training

import inspect

def print_line_number():
    line_number = inspect.currentframe().f_back.f_lineno
    #print({line_number})
    return line_number

print_line_number()

print("Code explanation")

dataframe = pd.read_csv('/content/drive/MyDrive/NLPassignment/Question1/bbc-text.csv')


class_labels = dataframe['category'].unique()


# Map the unique categories to integers using pd.factorize() method
category_map = dict(zip(class_labels, range(len(class_labels))))

# Map the categories to integers using the map() method
dataframe['category'] = dataframe['category'].map(category_map)


#number of class labels
num_classes=len(dataframe['category'].unique())

dataframe['text'],labels=preprocess.main(dataframe)


training_set, testing_set, train_class_labels, test_class_labels = train_test_split(dataframe['text'],dataframe['category'], test_size=0.2, random_state=42)

training.main(training_set, testing_set, train_class_labels, test_class_labels,dataframe,num_classes)
