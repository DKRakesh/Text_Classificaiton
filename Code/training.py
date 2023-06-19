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
from nnmodel import neuralmodel


def test(testing_set, test_class_labels,model):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(testing_set, dtype=torch.float32))
        _, predicted = torch.max(outputs.data, 1)


    accuracy = accuracy_score(test_class_labels, predicted)


    print('Accuracy: {:.2f}%'.format(accuracy * 100))
    
def main(training_set, testing_set, train_class_labels, test_class_labels,labels,num_classes):
    
    vectorizer = CountVectorizer()
    training_set = vectorizer.fit_transform(training_set).toarray()
    testing_set = vectorizer.transform(testing_set).toarray()
    
    input_size = training_set.shape[1]
    
    hidden_size = 10
    num_classes = num_classes
    learning_rate = 0.10
    num_epochs = 1
    

    model = neuralmodel(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(zip(training_set, train_class_labels)):
            inputs = torch.tensor(inputs, dtype=torch.float32)
            #print("inputs",inputs)
            labels = torch.tensor(labels, dtype=torch.long)
            #print("labels",labels)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs.unsqueeze(0), labels.unsqueeze(0))
            
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(training_set), loss.item()))

    test(testing_set, test_class_labels,model)