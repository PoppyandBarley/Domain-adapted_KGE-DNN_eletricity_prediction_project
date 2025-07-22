# -*- coding: utf-8 -*-
# @Time    : 2025-02-27 16:06
# @Author  : Antonio

import torch.nn as nn
import torch
from utils import linear_layer


class CompanyOperationEvaluation(nn.Module):
    def __init__(self, entity_num, head_num, relation_num, feature_dim, embed_dim, hidden_layers, dropouts, output_dim=1):
        super(CompanyOperationEvaluation, self).__init__()
        
        # Company feature processing
        self.feature_layer = linear_layer(feature_dim, embed_dim)
        
        # Knowledge graph embeddings
        self.head_embed = nn.Embedding(head_num, embed_dim)
        self.entity_embed = nn.Embedding(entity_num, embed_dim)
        self.relation_embed = nn.Embedding(relation_num, embed_dim)
        
        # Cross-compression weights
        self.compress_weight_cf = torch.rand((embed_dim, 1), requires_grad=True)
        self.compress_weight_ef = torch.rand((embed_dim, 1), requires_grad=True)
        self.compress_weight_fc = torch.rand((embed_dim, 1), requires_grad=True)
        self.compress_weight_fe = torch.rand((embed_dim, 1), requires_grad=True)
        self.compress_bias_c = torch.rand(1, requires_grad=True)
        self.compress_bias_e = torch.rand(1, requires_grad=True)
        
        #mlp for lower layer
        self.user_low_mlp_layer = linear_layer(embed_dim, embed_dim, dropout=0.5)
        self.relation_low_mlp_layer = linear_layer(embed_dim, embed_dim, dropout=0.5)

        # MLP for knowledge graph
        self.kg_layers = nn.Sequential()
        kg_layers = [2 * embed_dim] + hidden_layers
        for i in range(len(kg_layers) - 1):
            self.kg_layers.add_module(
                f'kg_hidden_layer_{i+1}',
                linear_layer(kg_layers[i], kg_layers[i+1], dropouts[i])
            )
        self.kg_layers.add_module('kg_last_layer', linear_layer(kg_layers[-1], embed_dim))
        
        # MLP for company operation prediction
        self.prediction_layers = nn.Sequential()
        pred_layers = [2 * embed_dim] + hidden_layers
        for i in range(len(pred_layers) - 1):
            self.prediction_layers.add_module(
                f'pred_hidden_layer_{i+1}',
                linear_layer(pred_layers[i], pred_layers[i+1], dropouts[i])
            )
        self.prediction_layers.add_module('pred_last_layer', linear_layer(pred_layers[-1], output_dim))
        self.softmax = nn.Softmax(dim=1)

    def cross_compress_unit(self, company_features, entity_embed):
        company_features_reshaped = company_features.unsqueeze(-1)
        entity_embed_reshaped = entity_embed.unsqueeze(-1)
        
        c = company_features_reshaped * entity_embed_reshaped.permute(0, 2, 1)
        c_t = entity_embed_reshaped * company_features_reshaped.permute(0, 2, 1)
        
        company_embed_c = torch.matmul(c, self.compress_weight_cf).squeeze() + \
                          torch.matmul(c_t, self.compress_weight_fc).squeeze() + self.compress_bias_c
        entity_embed_c = torch.matmul(c, self.compress_weight_ef).squeeze() + \
                         torch.matmul(c_t, self.compress_weight_fe).squeeze() + self.compress_bias_e
        return company_embed_c, entity_embed_c

    def forward(self, data, train_type):
        if train_type == 'rec':
            # rec module
            company_features = self.feature_layer(data[0].float())  # Process basic company features
            entity_embed = self.entity_embed(data[1].long())    # Related entity embeddings
            head_embed = self.head_embed(data[1].long())  # Relation embeddings
            operation_target = data[2].long()
            
            for _ in range(2):  # Two layers of cross compression
                company_features = self.user_low_mlp_layer(company_features)
                head_embed, entity_embed = self.cross_compress_unit(head_embed, entity_embed)
            
            high_layer = torch.cat((company_features, entity_embed), dim=1)
            operation_rank = self.prediction_layers(high_layer)
            operation_prob = self.softmax(operation_rank)
            
            return operation_prob, operation_target
        else:
            # kg module
            print(f"Data[1] range: min={data[1].min()}, max={data[1].max()}, relation_num={self.relation_embed.num_embeddings}")
            head_embed = self.head_embed(data[0].long())
            entity_embed = self.entity_embed(data[0].long())
            relation_embed = self.relation_embed(data[1].long())
            tail_embed = self.head_embed(data[2].long())

            for i in range(2):
                entity_embed, head_embed = self.cross_compress_unit(entity_embed, head_embed)
                relation_embed = self.relation_low_mlp_layer(relation_embed)
            high_layer = torch.cat((head_embed, relation_embed), dim=1)
            tail_out = self.kg_layers(high_layer)

            return tail_out, tail_embed
        
    
if __name__ == '__main__':

    # Define input parameters
    batch_size = 4
    feature_dim = 10  # Number of company basic features (size, revenue, etc.)
    company_num = 100  # Number of companies in dataset
    entity_num = 50  # Number of entities in knowledge graph
    head_num = 50
    relation_num = 20  # Number of relation types
    embed_dim = 16  # Embedding dimension
    hidden_layers = [32, 16]  # Hidden layers for MLP
    dropouts = [0.5, 0.5]

    # Instantiate the model
    model = CompanyOperationEvaluation(
        entity_num=entity_num+1,
        head_num=head_num+1,
        relation_num=relation_num+1,
        feature_dim=feature_dim,
        embed_dim=embed_dim,
        hidden_layers=hidden_layers,
        dropouts=dropouts,
        output_dim=3
    )

    # Generate random test input
    company_features = torch.rand(batch_size, feature_dim)  # Random company features
    entity_indices = torch.randint(0, entity_num, (batch_size,))  # Random entity IDs
    head_indices = torch.randint(0, head_num, (batch_size,))
    relation_indices = torch.randint(0, relation_num, (batch_size,))  # Random relation IDs
    tail_indices = torch.randint(0, head_num, (batch_size,))
    operation_target = torch.randint(1, 3, (batch_size,))

    # Forward pass
    data = (company_features, entity_indices, operation_target)
    kg = (head_indices, relation_indices, tail_indices)
    operation_prob, target = model(data, train_type='rec')

    # Print output
    print("Predicted Operation Rank:", operation_prob)
    print("Target Operation:", target)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training step
    optimizer.zero_grad()
    loss = criterion(operation_prob, target)
    loss.backward()
    optimizer.step()
