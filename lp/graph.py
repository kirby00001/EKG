import dgl
import torch
import torch.nn.functional as F

import neo4j.ap as ap
import neo4j.law as law
import neo4j.relation as rel
import neo4j.enterprise as enterprise


def construct_metagraph():
    """
    添加节点和边

    :return: metagraph
    """
    graph = dgl.heterograph(data_dict={
        ("公司", "公司持股", "公司"): rel.get_ete(),
        ("持股人", "个人持股", "公司"): rel.get_ets(),
        ("公司", "行政处罚", "行政处罚记录"): rel.get_eta(),
        ("公司", "当事人", "法律文书"): rel.get_etl("当事人"),
        ("公司", "原告", "法律文书"): rel.get_etl("原告"),
        ("公司", "被告", "法律文书"): rel.get_etl("被告"),
        ("公司", "第三人", "法律文书"): rel.get_etl("第三人"),
        ("公司", "上诉人", "法律文书"): rel.get_etl("上诉人"),
        ("公司", "被上诉人", "法律文书"): rel.get_etl("被上诉人"),
        ("公司", "申请执行人", "法律文书"): rel.get_etl("申请执行人"),
        ("公司", "被执行人", "法律文书"): rel.get_etl("被执行人"),
        ("公司", "其他", "法律文书"): rel.get_etl("其他"),
    })
    return graph


def _add_enterprise_data(graph):
    """
    '公司类型', '成立日期', '所属区县', '所属城市', '所属省份', '注册资本', '经营状态', '行业类型'


    :param graph: DGLgraph
    :return: DGLgraph with enterprise data
    """
    attributes = ['公司类型', '成立日期', '所属区县', '所属城市', '所属省份', '注册资本', '经营状态', '行业类型']
    data = []
    for attribute in attributes:
        data.append(torch.tensor(enterprise.get_enterprise_data(attribute)).unsqueeze(dim=-1))
    data = torch.cat(data, dim=1)
    graph.nodes["公司"].data["features"] = data


def _add_law_data(graph):
    """
    '日期', '案由'

    :param graph: DGLgraph
    :return: DGLgraph with enterprise data
    """
    attributes = ['日期', '案由']
    data = []
    for attribute in attributes:
        data.append(torch.tensor(law.get_law_data(attribute)).unsqueeze(dim=-1))
    data = torch.cat(data, dim=1)
    graph.nodes["法律文书"].data["features"] = data


def _add_ap_data(graph):
    """
    '决定日期', '决定机关', '行政处罚类型'

    :param graph: DGLgraph
    :return: DGLgraph with enterprise data
    """
    attributes = ['决定日期', '决定机关', '行政处罚种类']
    data = []
    for attribute in attributes:
        data.append(torch.tensor(ap.get_ap_data(attribute)).unsqueeze(dim=-1))
    data = torch.cat(data, dim=1)
    graph.nodes["行政处罚记录"].data["features"] = data


def _add_person_data(graph):
    """
    ''

    :param graph: DGLgraph
    :return: DGLgraph with enterprise data
    """
    graph.nodes["持股人"].data["features"] = graph.nodes("持股人").clone().detach().unsqueeze(dim=-1)


def add_node_data(graph):
    _add_enterprise_data(graph)
    _add_person_data(graph)
    _add_law_data(graph)
    _add_ap_data(graph)


def add_personal_stock_edge_data(graph):
    """
    '持股比例', '认缴出资日期', '认缴出资额'

    :param graph: DGLgraph
    :return: DGLgraph with 股东持股关系数据
    """
    attributes = ['持股比例', '认缴出资日期', '认缴出资额']
    data = []
    for attribute in attributes:
        data.append(torch.tensor(rel.get_ets_data(attribute)).unsqueeze(dim=-1))
    data = torch.cat(data, dim=1)
    graph.edges["个人持股"].data["features"] = data


def add_enterprise_stock_edge_data(graph):
    """
    '持股比例', '认缴出资日期', '认缴出资额'

    :param graph: DGLgraph
    :return: DGLgraph with 股东持股关系数据
    """
    attributes = ['持股比例', '认缴出资日期', '认缴出资额']
    data = []
    for attribute in attributes:
        data.append(torch.tensor(rel.get_ete_data(attribute)).unsqueeze(dim=-1))
    data = torch.cat(data, dim=1)
    graph.edges["公司持股"].data["features"] = data


def get_featureSize(graph):
    return [features.shape[-1] for features in graph.ndata['features'].values()]


def padding_to_sameSize(graph):
    for ntype in graph.ntypes:
        features = graph.nodes[ntype].data['features']
        size = features.shape[-1]
        graph.nodes[ntype].data['features'] = F.pad(features, pad=[0, 8 - size])


def get_graph():
    """
    获取DGLgraph

    :return: DGLgraph
    """
    # 建图
    graph = construct_metagraph()
    # 添加节点数据
    add_node_data(graph)
    # 添加边数据
    add_enterprise_stock_edge_data(graph)
    add_personal_stock_edge_data(graph)
    # 填充节点数据到同一大小
    padding_to_sameSize(graph)

    return graph


if __name__ == "__main__":
    # graph = get_graph()
    # print(graph)
    # dgl.save_graphs(filename="../data/DGLgraph", g_list=graph)
    graph, _ = dgl.load_graphs("../data/DGLgraph")
    graph = graph[0]
    print(graph)
    print(graph.ndata)
