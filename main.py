# -*- coding: utf-8 -*-
# @Time    : 2025-02-15 09:28
# @Author  : Antonio
# @Description :  script description


from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from mkr_model import CompanyOperationEvaluation
from train_model import KG_TrainSet, REC_TrainSet, train_model
from utils import load_data, load_kg
import argparse


def main(args):
    n_feature, n_entity, train_rec, eval_rec, test_rec = load_data()
    n_head, n_relation, kg = load_kg()

    kg_data = (kg.iloc[:, 0], kg.iloc[:, 1], kg.iloc[:, 2])
    rec_data = train_rec
    rec_val = eval_rec

    train_data_kg = KG_TrainSet(kg_data)
    train_loader_kg = DataLoader(train_data_kg, batch_size=args.batch_size, shuffle=args.shuffle_train)

    train_data_rec = REC_TrainSet(rec_data)
    eval_data_rec = REC_TrainSet(rec_val)

    train_loader_rec = DataLoader(train_data_rec, batch_size=args.batch_size, shuffle=args.shuffle_train)
    eval_loader_rec = DataLoader(eval_data_rec, batch_size=args.batch_size, shuffle=args.shuffle_test)

    model = CompanyOperationEvaluation(n_entity + 1, n_head + 1, n_relation + 1, n_feature,
                    embed_dim=args.batch_size,
                    hidden_layers=args.hidden_layers,
                    dropouts=args.dropouts, output_dim=args.output_rec)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    loss_function = nn.CrossEntropyLoss()
    epochs = args.epochs
    train_model(model, train_loader_rec, train_loader_kg, eval_loader_rec,
                optimizer, loss_function, epochs)


if __name__ == '__main__':
    # add argument
    parser = argparse.ArgumentParser(description="mkr model arguments")
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--shuffle_train", type=bool, default=True)
    parser.add_argument("--shuffle_test", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--output_rec", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_layers", nargs='+', type=int, default=[64, 64])
    parser.add_argument("--dropouts", nargs='+', type=float, default=[0.5, 0.5])
    args = parser.parse_args()
    main(args)
