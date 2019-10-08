import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import os
import numpy as np
import time
import random
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        # 因为是有监督的方式，所以最后还需要把Embedding转换为类别标签，因此需要训练一个weight来转换。
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        """传入nodes
        1.先用enc对nodes处理成embeds
        2.然后weight与embeds做矩阵乘法
        3.返回转置后的结果
        """
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        """传入nodes，labels
        1.先用用前向传播计算出结果scores
        2.然后用交叉熵算出结果scores与目标labels之间的代价
        """
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


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


def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
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

    feat_data, labels, adj_lists = load_cora()


    # 2708个节点，每个节点生成一个1433维的向量.
    # 对于每个节点的1433维上，用高斯分布进行初始化。存储在features.weight中。
    features = nn.Embedding(2708, 1433)
    # 用feat_data的值覆盖掉features.weight，且把features.weight转为可训练的状态
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()

    # 初始化agg1对象
    # 初始化传参 features = features，尺寸为(2708, 1433)
    # 在前向传播时，仅仅是对本次batch的每个节点，用其邻接点嵌入的平均来表示它自己，
    # 结果张量 neigh_feats 的尺寸为（batch_size,1433）
    agg1 = MeanAggregator(features, cuda=False)
    # 初始化enc1对象
    # 初始化传参 features = features，尺寸为(2708, 1433)，feature_dim=1433，embed_dim=128
    # 在前向传播时，仅仅相当于对 neigh_feats 进行了一层编码映射（矩阵乘法+激活），从1433维到128维
    # 结果张量 combined 的尺寸（128,batch_size）
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)

    # 初始化agg2对象
    # 初始化传参 features = enc1输出的转置，尺寸为（batch_size，128）
    # 在前向传播时，仅仅是对本次batch的每个节点，用其邻接点嵌入的平均来表示它自己，
    # 结果张量 neigh_feats 的尺寸为（batch_size,128）
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)

    # 初始化enc2对象
    # 初始化传参 features = enc1输出的转置，尺寸为（batch_size，128）
    # 在前向传播时，仅仅相当于对 neigh_feats 进行了一层编码映射（矩阵乘法+激活），从128维到128维
    # 结果张量 combined 的尺寸（128,batch_size）
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=False)

    enc1.num_samples = 5
    enc2.num_samples = 5

    # 通过SupervisedGraphSage类，生成一个完整的计算图，包括前向传播及损失计算
    # enc2可被看做一个几乎已串联完成的计算图，输入为features，输出为经过激活的combined
    graphsage = SupervisedGraphSage(7, enc2)
    #    graphsage.cuda()

    # 随机化一个节点的序列。即生成一个数组，长度为num_nodes，里面所有元素是对[0,num_nodes-1]一个序列的shuffle。
    rand_indices = np.random.permutation(num_nodes)
    """
    500个节点来测试
    2208个节点来训练
    """

    val = rand_indices[:500]
    train = rand_indices[500:]
    # 定义优化器
    # 其中filter与lambda的作用是：
    # 通过判断requires_grad的真假来过滤参数，即SGD只训练那些graphsage.parameters()里需要训练的参数（requires_grad为True）
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)

    for batch in range(120):
        # 每个batch只跑头256个节点，下一个batch的头256个节点又不一样（因为shuffle了）。
        batch_nodes = train[:256]
        random.shuffle(train)

        start_time = time.time()
        optimizer.zero_grad()
        # 计算损失
        # batch_nodes 是节点索引的数组
        # Variable(...) 是把batch_nodes 转换为节点对应label数组，格式为tensor
        loss = graphsage.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        # 计算梯度
        loss.backward()
        # 参数更新
        optimizer.step()


    train_output = graphsage.forward(train)
    # 用验证集的节点前向传播一次
    val_output = graphsage.forward(val)


    print("Train:", classification_report(labels[train], train_output.data.numpy().argmax(axis=1)))
    print("Validation:", classification_report(labels[val], val_output.data.numpy().argmax(axis=1)))
    # 打印f1_score，即认为precision和recall同等重要。
    # 因为val是验证集的样本的索引，所以labels[val]即可获得对应的label数组
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))




if __name__ == "__main__":
    run_cora()
