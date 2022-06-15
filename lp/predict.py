import dgl
import torch
import pandas as pd

from lp.lpmodel import LPmodel
from lp.training import add_reverse_hetero, get_num_nodes_dict


def construct_predict_graph(graph, etype):
    """
    生成测试数据

    :param graph: 实际图
    :param etype: 目标边的类型
    :return: 预测图
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
    }
    edge_dict = {}
    _, target_edge_type, _ = etype
    for edge_type in graph.etypes:
        # 对于目标边，不保留真实边，添加所有不存在的边
        if edge_type == target_edge_type:
            utype, _, vtype = etype
            # 真实边
            srcs, dsts = graph.edges(etype=etype)
            real_edges = list(zip(srcs, dsts))
            # 所有可能边
            possible_srcs = torch.arange(graph.number_of_nodes(utype) // 100).repeat_interleave(
                graph.number_of_nodes(vtype) // 100)
            possible_dsts = torch.arange(graph.number_of_nodes(vtype) // 100).repeat(
                graph.number_of_nodes(utype) // 100)
            possible_edges = list(zip(possible_srcs, possible_dsts))
            # 去除真实边
            predict_edges = list(set(possible_edges) - set(real_edges))
            # 提取节点
            predict_srcs = []
            predict_dsts = []
            for src, dst in predict_edges:
                predict_srcs.append(src)
                predict_dsts.append(dst)
            edge_dict[etype] = (torch.tensor(predict_srcs, dtype=torch.int),
                                torch.tensor(predict_dsts, dtype=torch.int))
        # 对于其他边，保留
        else:
            edge_dict[edge_map[edge_type]] = graph.edges(etype=edge_type)
    predict_graph = dgl.heterograph(data_dict=edge_dict, num_nodes_dict=get_num_nodes_dict(graph))
    # copy_ndata:
    for ntype in graph.ntypes:
        for k, v in graph.nodes[ntype].data.items():
            predict_graph.nodes[ntype].data[k] = v.detach().clone()
    return predict_graph


if __name__ == "__main__":
    # 载入实际图
    real_graph, _ = dgl.load_graphs("/home/kirby/pythonProject/EKG/data/DGLgraph")
    real_graph = real_graph[0]
    # 构造预测图
    predict_graph = construct_predict_graph(real_graph, ('持股人', '个人持股', '公司'))
    # 添加反向边
    reverse_predict_graph = add_reverse_hetero(predict_graph)
    print(reverse_predict_graph)
    # 载入模型
    model = LPmodel(8, 20, 8, reverse_predict_graph.etypes)
    model_parameters = torch.load("/home/kirby/pythonProject/EKG/model/BestModel.pth")
    model.load_state_dict(model_parameters)
    # CPU
    model = model.to("cuda")
    reverse_predict_graph = reverse_predict_graph.to("cuda")
    # 预测
    node_features = reverse_predict_graph.ndata["features"]
    scores = model.predict(reverse_predict_graph, node_features, ('持股人', '个人持股', '公司'))
    # 构造结果
    srcs, dsts = predict_graph.edges(etype=('持股人', '个人持股', '公司'))
    table = pd.DataFrame({
        "持股人": srcs.to('cpu').numpy(),
        "公司": dsts.to('cpu').numpy(),
        "分数": scores.to('cpu').detach().numpy(),
    })
    print(table)
    table.to_csv("/home/kirby/pythonProject/EKG/data/link.csv", index=False)
