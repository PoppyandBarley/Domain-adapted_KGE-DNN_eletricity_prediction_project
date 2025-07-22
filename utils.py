# -*- coding: utf-8 -*-
# @Time    : 2025-02-15 09:28
# @Author  : Antonio
# @Description :  script description

import torch.nn as nn
import torch
import os
import numpy as np
import pandas as pd


def linear_layer(input, output, dropout=0):
    """
    linear layer for deep neural network model
    :param input: int, input layer unit
    :param output: int, output layer unit
    :param dropout: float, dropout ratio default 0
    :return: tensor
    """
    return nn.Sequential(
        nn.Linear(input, output),
        nn.LeakyReLU(),
        nn.Dropout(dropout)
    )


def dataset_split(rating_np):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    from sklearn.model_selection import train_test_split
    col_name = rating_np.columns.values.tolist()
    train_data, eval_test_data = train_test_split(rating_np, test_size=0.4, random_state=42)
    eval_data, test_data = train_test_split(eval_test_data, test_size=0.5, random_state=42)
    train_data = pd.DataFrame(train_data, columns=col_name)
    eval_data = pd.DataFrame(eval_data, columns=col_name)
    test_data = pd.DataFrame(test_data, columns=col_name)
    return train_data, eval_data, test_data

def load_data():
    print('reading rating file ...')

    # reading rating file
    rating_file = r".\comp_data_processed.xlsx"
    if os.path.exists(rating_file):
        rating_pd = pd.read_excel(rating_file)
    else:
        print('cannot find rating file')

    n_feature = rating_pd.shape[1] - 2
    n_entity = len(set(rating_pd['company']))
    train_data, eval_data, test_data = dataset_split(rating_pd)

    return n_feature, n_entity, train_data, eval_data, test_data
def load_kg():
    print('reading KG file ...')

    # reading kg file
    kg_file = r'.\knowledge_graph_new.csv'
    if os.path.exists(kg_file):
        kg = pd.read_csv(kg_file, header=None)
        
    n_head = len(set(kg.iloc[:, 0]) | set(kg.iloc[:, 2]))
    n_relation = len(set(kg.iloc[:, 1]))

    return n_head, n_relation, kg


def multi_loss(pred, target, types, loss_function):
    if types == "rec":
        loss = loss_function(pred, target)
        return loss
    else:

        loss = torch.sigmoid(torch.sum(pred * target))
        return loss