import networkx as nx
import numpy as np
import random
from collections import defaultdict
import os
from sklearn.metrics import roc_auc_score
import random
from gensim.models import Word2Vec
from sklearn.metrics import f1_score

path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class Node2Vec:
    def __init__(self, G, emb_size=128, p=4, q=1, length_walk=50, num_walks=10, window_size=10, num_iters=2):
        self.G = G
        self.emb_size = emb_size
        self.length_walk = length_walk
        self.num_walks = num_walks
        self.window_size = window_size
        self.num_iters = num_iters
        self.p = p
        self.q = q

    def walk_step(self, last_1th_node, last_2th_node):
        """
        主要是针对倒数第2个节点last_2th_node的邻接点群里，采样一个节点并返回。
        1.特殊情况处理
            last_2th_node是本次walk数组里保存的倒数第2个邻接点，
            如果它周边没有节点了就返回false，让walk数组停止扩充。

        2.备选权重修正
            last_1th_node是本次walk数组里保存的倒数第1个邻接点，
            如果last_2th_node的邻接点是离原始node更远（即last_1th_node），
            则该邻接点的被选权重调整为 1 / self.p

            如果last_2th_node的邻接点是离原始node更近（不是last_1th_node，也与last_1th_node不邻接），
            则该邻接点的被选权重调整为 1 / self.q

            如果last_2th_node的邻接点是与原始node的距离不变，
            则该邻接点的被选权重保持为 1

        3. 按权重采样。
        """

        nbs = list(self.G.neighbors(last_2th_node))
        # 1.特殊情况处理
        if len(nbs) == 0:
            return False

        # 2.备选权重修正
        # 所有节点的权重都为1
        weights = [1] * len(nbs)
        for i, x in enumerate(nbs):
            """
            如果节点是最后一个节点，
            该节点的权重就为1/p
            
            如果节点不是最后一个节点，且与最后一个节点之间没有连接，
            该节点的权重为1/q
            """
            if last_1th_node == x:
                weights[i] = 1 / self.p
            elif not self.G.has_edge(last_1th_node, x):
                weights[i] = 1 / self.q

        # 3. 按权重采样。
        # 根据更新后的权重，从last_2th_node节点的邻接节点里抽出一个节点。
        return random.choices(nbs, weights=weights, k=1)[0]

    def random_walk(self):
        """
        对图中每一个节点，对其邻居节点进行random_walk
        返回每个节点的邻接点数组的数组，相当于很多个句子
             walks = [['2','5',''41'],['7','3'],...]
        """
        walks = []
        """ 每一次 for node 循环，
        都是在求某个节点node的邻接点构成的数组walk 
        """
        for node in self.G.nodes():
            walk = [node]
            nbs = list(self.G.neighbors(node))
            if len(nbs) > 0:
                walk.append(random.choice(nbs))
                """ 每一次 for i 循环，都是为特定的节点node
                来扩充它的邻接点构成的数组walk
                最大长度length_walk提前设定。
                """
                for i in range(2, self.length_walk):
                    v = self.walk_step(walk[-1], walk[-2])
                    if not v:
                        break
                    walk.append(v)
            # 所有节点转成字符串
            walk = [str(x) for x in walk]
            # 每个节点有一个对应的字符串数组。
            walks.append(walk)

        return walks

    def sentenses(self):
        """
        1、调用random_walk方法来生成“一段话”，即图上每个节点对应的“句子”，
        2、多次重复这个调用，相当于扩充这个“语料”
        """
        sts = []
        for _ in range(self.num_walks):
            sts.extend(self.random_walk())
        return sts

    def train(self, workers=4, is_loadmodel=False, is_loaddata=False):
        # 如果有已训练好的Node2Vec模型，直接载入
        if is_loadmodel:
            print('Load model from file')
            w2v = Word2Vec.load(path + '/node2vec/Node2Vec.model')
            return w2v

        """
        如果有已预处理好的"句子"数据，就直接读取
        否则，生成"句子"数据，并保存
        """
        if is_loaddata:
            print('Load data from file')
            with open(path + '/node2vec/walk_node2vec.txt', 'r') as f:
                sts = f.read()
                sentenses = eval(sts)
        else:
            print('Random walk to get training data...')
            sentenses = self.sentenses()
            print('Number of sentenses to train: ', len(sentenses))
            with open(path + '/node2vec/walk_node2vec.txt', 'w') as f:
                f.write(str(sentenses))

        """
        以下才是真正的训练，即用上面生成的句子语料，喂给Word2Vec进行训练，结果为Node2Vec模型。
        """
        print('Start training...')
        random.seed(616)
        w2v = Word2Vec(sentences=sentenses, size=self.emb_size, window=self.window_size, iter=self.num_iters, sg=1,
                       hs=1, min_count=0, workers=workers)
        w2v.save(path + '/node2vec/Node2Vec.model')
        print('Training Done.')

        return w2v

def load_cora():
    """
    cora.content 是一个机器学习领域论文的数据表
    共2708行，对应2708篇论文
    共1435列，其中：
        第一列是每个论文的id，
        中间1433列，是一个词表的统计，表示1433个关键词的有无，如果有是1，没有是0.
        最后一列是每个论文的label（即7个子领域）

    cora.cites 是论文之间的引用数据表
    每行对应一个引用关系，有两列
    第一列是被引用的论文id，第二列是引用论文的id。相当于 右论文 指向 左论文

    """
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    # print(path)
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open(path + "/cora/cora.content") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            # print('info[1:-1]=',info[1:-1])
            feat_data[i, :] = list(map(float, info[1:-1]))

            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open(path + "/cora/cora.cites") as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists



def main():
    """仅用于单元测试"""
    random.seed(616)

    """
    feat_data 是每篇论文对应的词表
    是一个二维数组，共2708行，1433列。

    labels 是论文的类别标签，共分7个不同子领域 
    是一个二维数组，共2708行，1列。每行元素为一个节点对应的label,共7种，从0到6.

    adj_lists 论文引用关系表。
    字典的字典，里面共有2708个元素，每个元素的key为节点编号，值为该节点的邻接点组成的字典
    例如其中一个元素为 310: {2513, 74, 747, 668}
    由于监督任务是为了区分子领域，因此这个引用关系表被构造成无向图。
    """
    # 读取边数据
    _, _, adj_lists = load_cora()
    edges_set=set()
    for key in adj_lists:
        tmp_dic=adj_lists[key]
        for i in tmp_dic:
            tmp_tuple=(key,i)
            edges_set.add(tmp_tuple)
    edges_list=list(edges_set)

    # 构造图数据
    G = nx.Graph()
    for i in range(2708):
        G.add_node(i)
    G.add_edges_from(edges_list)

    node2vec = Node2Vec(G, emb_size=128, p=4, q=1, length_walk=50, num_walks=10, window_size=10, num_iters=2)
    w2v = node2vec.train(workers=4, is_loadmodel=False, is_loaddata=True)

    # 由于是无监督模型，需根据具体任务的指标进行测试。

if __name__ == '__main__':
    main()
