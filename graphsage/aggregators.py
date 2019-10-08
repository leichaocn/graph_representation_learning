import torch
import torch.nn as nn
from torch.autograd import Variable

import random

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """
        """
        features -- 保存所有节点的embedding，是二维数组。例如(2708, 1433)，即2708个节点，每个节点是1433维向量
        """
        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    # 这里MeanAggregator对象的forward方法，是在Encoder对象的forward方法调用，用于构造计算图。
    """
    需要传入三个参数，
    第一个元素nodes，是一个batch里的节点数组，
    第二个元素to_neighs，是所有节点的邻接点信息，格式为一个数组，元素索引对应nodes里的索引，元素值为其邻接节点的集合。
    第三个元素num_sample为采样
    
    这里需要注意，nodes里的元素值已经失去了索引意义，例如：
    nodes = [213,4,35]
    由于 adj_lists = {4:{7,9},...,35:{49,20},...,213: {2513, 74, 747, 668},...}
    则 to_neighs = [{2513, 74, 747, 668},{7,9},{49,20}]
    """
    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set

        """
        如果num_sample有具体数字，就进行采样
        如果num_sample为None，就直接使用to_neighs
        """
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        """
        
        """
        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]

        # 把samp_neighs里所有元素值进行去重。
        # unique_nodes_list是一个数组，元素为所有邻接节点的索引。
        unique_nodes_list = list(set.union(*samp_neighs))

        # unique_nodes_list 转换为 unique_nodes
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}

        # 构建一个二维tensor，
        # 行数是本次前向传播的节点个数 samp_neighs
        # 列数是本次前向传播的节点们对应的邻接点总数 unique_nodes
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))


        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]

        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]

        mask[row_indices, column_indices] = 1

        """
        如果以上这部分看起来很抽象，那么跑一个小例子就明白了
        samp_neighs = [{1, 2}, {0}, {2, 10}]
        unique_nodes_list = [0, 1, 2, 10]
        unique_nodes = {0: 0, 1: 1, 2: 2, 10: 3}
        column_indices = [1, 2, 0, 2, 3]
        row_indices = [0, 0, 1, 2, 2]
        mask = [[0. 1. 1. 0.]
                [1. 0. 0. 0.]
                [0. 0. 1. 1.]]
        即目的是把邻接节点集合数组samp_neighs，转换为一个二维的mask，行对应本次batch里的节点，列对应该节点的邻接节点。
        而且进行了压缩，防止稀疏，通过字典unique_nodes的key里保存了节点的原始索引。
        """


        """
        计算均值
        dim=1，把每一行的所有元素加起来；keepdim=True，保持尺寸形状不变
        如果按照上面那个mask的小例子，则为
        num_neigh = [[2]
                [1]
                [2]]
        mask = [[0. 1/2 1/2 0.]
                [1. 0. 0. 0.]
                [0. 0. 1/2 1/2]]
        """
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)

        # embed_matrix是从二维张量 features 里，把所有邻接节点的嵌入表示拿出来
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))

        # 矩阵乘法
        # 用mask乘以embed_matrix，
        # （本batch节点个数，邻接节点个数）*（邻接节点个数，每个节点embedding维度）=（本batch节点个数，每个节点embedding维度）
        # 用上面的小例子，结果to_feats的shape就是（3,1433）
        to_feats = mask.mm(embed_matrix)
        return to_feats
