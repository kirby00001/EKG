import dgl
import torch
from dgl import data
import torch.nn.functional as F


class EKGDataset(data.DGLDataset):
    def __init__(self, name="", raw_dir="../data/"):
        self._g = None
        self._name = name
        super(EKGDataset, self).__init__(name, raw_dir=raw_dir)

    def process(self):
        g, _ = dgl.load_graphs(self.raw_path + "/DGLgraph")
        g = g[0]
        # 添加 etype
        for i, type in enumerate(g.etypes):
            g.edges[type].data["etype"] = torch.tensor(i).expand_as(g.edges(etype=type)[0])
        # 添加 ntype
        for i, type in enumerate(g.ntypes):
            g.nodes[type].data["ntype"] = torch.tensor(i).expand_as(g.nodes(type))
        self._g = g

    def __getitem__(self, index):
        assert index == 0
        return self._g

    def __len__(self):
        return 1





if __name__ == "__main__":
    # graph = EKGDataset()
    # print(graph[0].edges["个人持股"])
    # print(graph[0].nodes["公司"])
    # data.add_nodepred_split(graph, [0.8, 0.1, 0.1], ntype="公司")
    # print(graph[0].nodes["公司"])
    g, _ = dgl.load_graphs("../data/DGLgraph")
    g = g[0]
    dgl.save_graphs("../data/DGLgraph", g)
