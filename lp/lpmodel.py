import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.nn.pytorch as dglnn
import dgl.function.message as fn


class RGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, relation_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_features, hidden_features)
                                            for rel in relation_names}, aggregate='mean')

        self.conv2 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(hidden_features, out_features)
                                            for rel in relation_names}, aggregate='mean')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class HeteroDotProductPredictor(nn.Module):
    """
    Distmulti打分函数：点乘
    """

    def __init__(self):
        super(HeteroDotProductPredictor, self).__init__()

    def forward(self, graph, h, etype):
        with graph.local_scope():
            for ntype in graph.ntypes:
                graph.nodes[ntype].data['h'] = h[ntype]
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score'].squeeze()


class LPmodel(nn.Module):
    """
    由一个编码器和一个解码器组成
    编码器负责学习节点的隐藏特征
    解码器负责根据节点表示计算分数
    并使用损失函数评价结果
    """

    def __init__(self, in_features, hidden_features, out_features, relation_names):
        super(LPmodel, self).__init__()
        # Encoder RGCN or RGraphSAGE
        self.conv = RGCN(in_features, hidden_features, out_features, relation_names)
        # Decoder 计算分数
        self.score = HeteroDotProductPredictor()

    def forward(self, pos_graph, neg_graph, feature, etype):
        hidden_feature = self.conv(pos_graph, feature)
        pos_score = self.score(pos_graph, hidden_feature, etype)
        neg_score = self.score(neg_graph, hidden_feature, etype)
        return pos_score, neg_score

    def predict(self, pos_graph, feature, etype):
        hidden_feature = self.conv(pos_graph, feature)
        pos_score = self.score(pos_graph, hidden_feature, etype)
        return pos_score
