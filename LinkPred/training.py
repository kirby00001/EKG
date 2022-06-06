import dgl
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from LinkPred.lpmodel import LPmodel


def get_num_nodes_dict(graph):
    return {ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}


def add_reverse_hetero(graph):
    relations = {}
    for metapath in graph.canonical_etypes:
        # Original edges
        src, dst = graph.all_edges(etype=metapath[1])
        relations[metapath] = (src, dst)
        # Reverse edges
        relations[(metapath[2], metapath[1] + "_by", metapath[0])] = (dst, src)

    new_g = dgl.heterograph(relations, num_nodes_dict=get_num_nodes_dict(graph))
    # copy_ndata:
    for ntype in graph.ntypes:
        for k, v in graph.nodes[ntype].data.items():
            new_g.nodes[ntype].data[k] = v.detach().clone()
    # copy_ndata:
    for etype in graph.etypes:
        for k, v in graph.edges[etype].data.items():
            new_g.edges[etype].data[k] = v.detach().clone()
            new_g.edges[etype + "_by"].data[k] = v.detach().clone()
    return new_g


def split_train_test(graph, test, target_edge_type):
    train_relations = {}
    test_relations = {}
    for metapath in graph.canonical_etypes:
        if metapath == target_edge_type:  # Target edges
            src, dst = graph.all_edges(etype=metapath[1])
            eids = graph.all_edges(form='eid', etype=metapath[1])
            test_size = int(test * len(eids))
            test_relations[metapath] = (src[:test_size], dst[:test_size])
            train_relations[metapath] = (src[test_size:], dst[test_size:])
        else:  # Other edges
            src, dst = graph.all_edges(etype=metapath[1])
            train_relations[metapath] = (src, dst)
            test_relations[metapath] = (src, dst)
    train_graph = dgl.heterograph(train_relations, num_nodes_dict=get_num_nodes_dict(graph))
    test_graph = dgl.heterograph(test_relations, num_nodes_dict=get_num_nodes_dict(graph))
    # copy_ndata:
    for ntype in graph.ntypes:
        for k, v in graph.nodes[ntype].data.items():
            train_graph.nodes[ntype].data[k] = v.detach().clone()
            test_graph.nodes[ntype].data[k] = v.detach().clone()
    # copy_ndata:
    for edge_type in graph.canonical_etypes:
        if edge_type == target_edge_type:
            continue
        else:
            for k, v in graph.edges[edge_type[1]].data.items():
                train_graph.edges[edge_type[1]].data[k] = v.detach().clone()
                test_graph.edges[edge_type[1]].data[k] = v.detach().clone()
    return train_graph, test_graph


def construct_negative_graph(graph, k, etype):
    """
    负采样

    :param graph:
    :param k:
    :param etype:
    :return:
    """
    edge_map = {
        "公司持股": ("公司", "公司持股", "公司"),
        "个人持股": ("持股人", "个人持股", "公司"),
        "行政处罚": ("公司", "行政处罚", "行政处罚记录"),
        "当事人": ("公司", "当事人", "法律文书"),
        "原告": ("公司", "原告", "法律文书"),
        "被告": ("公司", "被告", "法律文书"),
        "第三人": ("公司", "第三人", "法律文书"),
        "上诉人": ("公司", "上诉人", "法律文书"),
        "被上诉人": ("公司", "被上诉人", "法律文书"),
        "申请执行人": ("公司", "申请执行人", "法律文书"),
        "被执行人": ("公司", "被执行人", "法律文书"),
        "其他": ("公司", "其他", "法律文书"),
        "公司持股_by": ("公司", "公司持股", "公司"),
        "个人持股_by": ("公司", "个人持股", "持股人"),
        "行政处罚_by": ("行政处罚记录", "行政处罚", "公司"),
        "当事人_by": ("法律文书", "当事人", "公司"),
        "原告_by": ("法律文书", "原告", "公司"),
        "被告_by": ("法律文书", "被告", "公司"),
        "第三人_by": ("法律文书", "第三人", "公司"),
        "上诉人_by": ("法律文书", "上诉人", "公司"),
        "被上诉人_by": ("法律文书", "被上诉人", "公司"),
        "申请执行人_by": ("法律文书", "申请执行人", "公司"),
        "被执行人_by": ("法律文书", "被执行人", "公司"),
        "其他_by": ("法律文书", "其他", "公司"),
    }
    edge_dict = {}
    _, target_edge_type, _ = etype
    for edge_type in graph.etypes:
        if edge_type == target_edge_type:
            utype, _, vtype = etype
            src, dst = graph.edges(etype=etype)
            neg_src = src.repeat_interleave(k)
            neg_dst = torch.randint(0, graph.number_of_nodes(vtype), (len(src) * k,)).to("cuda")
            edge_dict[etype] = (neg_src, neg_dst)
        else:
            edge_dict[edge_map[edge_type]] = graph.edges(etype=edge_type)
    return dgl.heterograph(data_dict=edge_dict, num_nodes_dict=get_num_nodes_dict(graph))


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).to('cpu').numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)


def cross_entropy_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to('cuda')
    return F.binary_cross_entropy_with_logits(scores, labels)


def training(model, optimizer, graph, etype, k):
    node_features = graph.ndata["features"]
    negative_graph = construct_negative_graph(graph, k, etype)
    pos_score, neg_score = model(graph, negative_graph, node_features, etype)
    loss = cross_entropy_loss(pos_score, neg_score)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        auc = compute_auc(pos_score, neg_score)
    return loss.item(), auc


def testing(model, graph, etype, k):
    with torch.no_grad():
        node_features = graph.ndata["features"]
        negative_graph = construct_negative_graph(graph, k, etype)
        pos_score, neg_score = model(graph, negative_graph, node_features, etype)
        loss = cross_entropy_loss(pos_score, neg_score)
        auc = compute_auc(pos_score, neg_score)
    return loss.item(), auc


if __name__ == "__main__":
    # 加载数据
    graph, _ = dgl.load_graphs("../data/DGLgraph")
    graph = graph[0]
    # 划分数据
    train_graph, test_graph = split_train_test(graph, test=0.2, target_edge_type=('持股人', '个人持股', '公司'))
    # 添加反向边
    train_graph = add_reverse_hetero(train_graph)
    test_graph = add_reverse_hetero(test_graph)
    # 训练模型
    model = LPmodel(8, 20, 8, train_graph.etypes).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 5000
    best_AUC = 0
    for epoch in range(0, epochs):
        train_loss, train_auc = training(model, optimizer, train_graph.to('cuda'), ('持股人', '个人持股', '公司'), k=1)
        test_loss, test_auc = testing(model, test_graph.to('cuda'), ('持股人', '个人持股', '公司'), k=1)
        if epoch % 100 == 0:
            print(
                f"epoch:{epoch}/{epochs}\n"
                f"  Train Error: loss:{train_loss:.5f} AUC:{train_auc:.5f}\n"
                f"  Test  Error: loss:{test_loss:.5f} AUC:{test_auc:.5f}")
        # 保存最优模型
        if best_AUC < test_auc:
            torch.save(obj=model.state_dict(), f=f"../model/BestModel.pth")
            best_AUC = test_auc
            print(f"Best Model: AUC:{best_AUC}")
    # 预测
