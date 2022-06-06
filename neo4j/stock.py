from neo4j import neo4j


def query_shareholder(enterprise):
    """
    查询企业的个人股东和持股比例
    :param enterprise:企业标识符
    :return:企业的个人股东和持股比例
    """
    result = neo4j.run(
        f'MATCH p=(e:E_Entity)-[r:`股东持股`]->(s:S_Entity) '
        f'WHERE e.p_pid="{enterprise}" '
        f'RETURN  s.`股东` AS name, r.`持股比例` AS rate ')
    result = result.to_data_frame()
    return result


def query_stock_enterprise(enterprise):
    """
    查询企业的公司股东
    :param enterprise:企业标识符
    :return:企业的公司股东和持股比例
    """
    result = neo4j.run(
        f'MATCH p=(e:E_Entity)-[r:`股东公司持股`]->(holder:E_Entity) '
        f'WHERE e.p_pid="{enterprise}" '
        f'RETURN  holder.p_pid AS pid, holder.`企业名称` AS name, r.`持股比例` AS rate')
    result = result.to_data_frame()
    return result


def query_stock_structure(enterprise, path):
    """
    查询指定企业的股权结构
    :param enterprise:企业标识符
    :param path: 路径
    :return:
    """
    result = []
    # 股东持股
    shareholders = query_shareholder(enterprise).to_dict("records")
    for shareholder in shareholders:
        shareholder["type"] = "people"
        shareholder["list"] = []
        result.append(shareholder)
    # 企业持股
    stock_enterprises = query_stock_enterprise(enterprise).to_dict("records")
    for stock_enterprise in stock_enterprises:
        pid = stock_enterprise["pid"]
        if pid not in path:
            path.append(pid)
            stock_enterprise["type"] = "enterprise"
            stock_enterprise["list"] = query_stock_structure(pid, path)
            result.append(stock_enterprise)
    return result


if __name__ == "__main__":
    pid = "97322340628206"
    print(query_stock_structure(pid, [pid]))
    # query_shareholder(pid)
    # query_stock_enterprise(pid)
