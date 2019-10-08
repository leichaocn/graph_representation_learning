import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists, aggregator,
            num_sample=10,
            base_model=None, gcn=False, cuda=False, 
            feature_transform=False): 
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        """
        gcn与feat_dim决定了本层输入的维度，embed_dim 决定了本层输出的维度。
        如果gcn为True，则weight的shape为（embed_dim，feat_dim）
        对于 enc1，embed_dim=128，feat_dim=1433，则weight的shape为（128,1433）
        对于 enc2，embed_dim=128，feat_dim=128，则weight的shape为（128,128）
        
        如果gcn为False，则weight的shape为（embed_dim，2*feat_dim）
        相当于W矩阵与B矩阵，在列上堆叠起来起来。
        对于 enc1，embed_dim=128，feat_dim=1433，则weight的shape为（128,2866）
        对于 enc2，embed_dim=128，feat_dim=128，则weight的shape为（128,256）

        """
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        """
        aggregator.forward的作用是构造aggregator阶段的计算图
        需要传入三个参数，
        第一个元素nodes，是一个batch里的节点数组，
        第二个元素to_neighs，是所有节点的邻接点信息，格式为一个数组，元素索引对应nodes里的索引，元素值为其邻接节点的集合。
        第三个元素num_sample为采样
        
        这里需要注意，nodes里的元素值已经失去了索引意义，例如：
        nodes = [213,4,35]
        由于 adj_lists = {4:{7,9},...,35:{49,20},...,213: {2513, 74, 747, 668},...}
        则 to_neighs = [{2513, 74, 747, 668},{7,9},{49,20}]
        
        获得的 neigh_feats 是本batch内所有节点的 邻接节点embedding的平均值。
        neigh_feats 的shape为（batch_size,1433）
        """
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], 
                self.num_sample)

        """
        这一步，是是否考虑自己的关键
        gcn=True，不考虑自己本层的embedding，相当于没B矩阵
        gcn=False,考虑自己本层的embedding，相当于有B矩阵
        """
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            """
            把 self_feats 与 neigh_feats，在列上拼接起来。
            
            对于enc1来说，
            self_feats 的尺寸为 （batch_size，1433）
            neigh_feats 的尺寸为 （batch_size，1433）
            combined 的尺寸为 （batch_size，2866）
            
            对于enc2来说，
            self_feats 的尺寸为 （batch_size，128）
            neigh_feats 的尺寸为 （batch_size，128）
            combined 的尺寸为 （batch_size，256）
            
            """
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats

        """
        共4中情况
        如果gcn为True，则weight的shape为（embed_dim，feat_dim）
        对于 enc1，embed_dim=128，feat_dim=1433，则weight的shape为（128,1433），
        由于combined=neigh_feats 的shape为（batch_size,1433），combined.t().shape 为（1433,batch_size）
        相乘再激活后，结果 combined.shape 为（128,batch_size）
        
        对于 enc2，embed_dim=128，feat_dim=128，则weight的shape为（128,128）
        由于combined=neigh_feats 的shape为（batch_size,128），combined.t().shape 为（128,batch_size）
        相乘再激活后，结果 combined.shape 为（128,batch_size）
        
        
        如果gcn为False，则weight的shape为（embed_dim，2*feat_dim）
        对于 enc1，embed_dim=128，feat_dim=1433，则weight的shape为（128,2866）
        由于combined 的尺寸为 （batch_size，2866），combined.t().shape 为（2866,batch_size）
        相乘再激活后，结果 combined.shape 为（128,batch_size）
        
        对于 enc2，embed_dim=128，feat_dim=128，则weight的shape为（128,256）
        由于combined 的尺寸为 （batch_size，256），combined.t().shape 为（256,batch_size）
        相乘再激活后，结果 combined.shape 为（128,batch_size）
        """
        combined = F.relu(self.weight.mm(combined.t()))
        return combined
