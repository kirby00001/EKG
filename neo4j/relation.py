import pandas as pd
import torch
from py2neo import Graph

from neo4j import neo4j


def get_ete():
    result = neo4j.run("MATCH p=(e1:E_Entity)-[r:`股东公司持股`]->(e2:E_Entity) RETURN id(e1) AS 被持股公司, id(e2) AS 持股公司")
    result = result.to_data_frame().to_dict(orient="list")
    return torch.tensor(result["持股公司"]), torch.tensor(result["被持股公司"])


def get_ets():
    mapper = neo4j.run("MATCH (n:S_Entity) RETURN id(n) AS id").to_data_frame()
    mapper["持股人"], _ = pd.factorize(mapper["id"])
    mapper.index = mapper["id"]

    result = neo4j.run("MATCH p=(e:E_Entity)-[r:`股东持股`]->(s:S_Entity) RETURN id(e) AS 被持股公司, id(s) AS 持股人")
    result = result.to_data_frame()
    result['持股人'] = result['持股人'].map(lambda x: mapper.loc[x, "持股人"])
    result = result.to_dict(orient="list")
    return torch.tensor(result["持股人"]), torch.tensor(result["被持股公司"])


def get_eta():
    mapper = neo4j.run("MATCH (n:AP_Entity) RETURN id(n) AS id").to_data_frame()
    mapper["行政处罚记录"], _ = pd.factorize(mapper["id"])
    mapper.index = mapper["id"]

    result = neo4j.run("MATCH p=(e:E_Entity)-[r:`有行政处罚`]->(ap:AP_Entity) RETURN id(e) AS 公司, id(ap) AS 行政处罚记录")
    result = result.to_data_frame()
    result['行政处罚记录'] = result['行政处罚记录'].map(lambda x: mapper.loc[x, "行政处罚记录"])
    result = result.to_dict(orient="list")
    return torch.tensor(result["公司"]), torch.tensor(result["行政处罚记录"])


def get_etl(relationship):
    mapper = neo4j.run("MATCH (n:LL_Entity) RETURN id(n) AS id").to_data_frame()
    mapper["法律文书"], _ = pd.factorize(mapper["id"])
    mapper.index = mapper["id"]

    result = neo4j.run(
        f"MATCH p=(e:E_Entity)-[r:`{relationship}`]->(l:LL_Entity) RETURN id(e) AS 公司, id(l) AS 法律文书")
    result = result.to_data_frame()
    result["法律文书"] = result["法律文书"].map(lambda x: mapper.loc[x, "法律文书"])
    result = result.to_dict('list')
    return torch.tensor(result["公司"]), torch.tensor(result["法律文书"])


def get_ete_data(attr):
    result = neo4j.run(f"MATCH p=(e1:E_Entity)-[r:`股东公司持股`]->(e2:E_Entity) RETURN r.`{attr}` AS data")
    result = result.to_data_frame()
    result["data"], _ = pd.factorize(result['data'])
    return result.to_dict(orient="list")['data']


def get_ets_data(attr):
    result = neo4j.run(f"MATCH p=(e:E_Entity)-[r:`股东持股`]->(s:S_Entity) RETURN r.`{attr}` AS data")
    result = result.to_data_frame()
    result["data"], _ = pd.factorize(result['data'])
    return result.to_dict(orient="list")['data']


if __name__ == "__main__":
    # print(get_ete())
    # print(get_ets())
    # print(get_eta())
    # print(get_etl("当事人"))
    # print(get_etl("原告"))
    # print(get_etl("被告"))
    # print(get_etl("第三人"))
    # print(get_etl("上诉人"))
    # print(get_etl("被上诉人"))
    # print(get_etl("申请执行人"))
    # print(get_etl("被执行人"))
    # print(get_etl("其他"))
    # print(torch.tensor(get_ete_data("持股比例")))
    print(torch.tensor(get_ets_data("持股比例")))
