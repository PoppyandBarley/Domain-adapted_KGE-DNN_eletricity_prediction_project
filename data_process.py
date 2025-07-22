# -*- coding: utf-8 -*-
# @Time    : 2025-02-15 11:01
# @Author  : Antonio
# @Description :  data_process

import argparse
import numpy as np
import pandas as pd

def convert_kg():
    print('converting kg.txt file ...')
    entity2index = pd.read_excel(r'c:\Users\hp\Desktop\广西电力项目\数据\entity2index.xlsx')
    entity_id2index = dict()
    for i in range(len(entity2index)):
        entity_id2index[entity2index.iloc[i, 0]] = entity2index.iloc[i, 1]
    entity_cnt = len(entity_id2index)
    relation_cnt = len(relation_id2index)

    writer = open(r'c:\Users\hp\Desktop\广西电力项目\数据\knowledge_graph_new.csv', 'w', encoding='utf-8')
    file = open(r'c:\Users\hp\Desktop\广西电力项目\数据\knowledge_graph.csv', encoding='utf-8')

    for line in file:
        array = line.strip().split(',')
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        if head_old not in entity_id2index:
            entity_id2index[head_old] = entity_cnt
            entity_cnt += 1
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]

        writer.write('%d,%d,%d\n' % (head, relation, tail))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)

if __name__ == '__main__':
    np.random.seed(555)
    relation_id2index = dict()

    convert_kg()

    print('done')
