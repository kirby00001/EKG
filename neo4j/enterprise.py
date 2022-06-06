import torch
import pandas as pd
import torch.nn.functional as F

from neo4j import neo4j


def query_all_enterprise():
    result = neo4j.run(
        f'MATCH (e:E_Entity)'
        f'RETURN e.p_pid AS pid')
    result = result.to_ndarray().squeeze()
    return result


def get_enterprise_data(attr):
    result = neo4j.run(f"MATCH (n:E_Entity) RETURN n.`{attr}` AS data")
    result = result.to_data_frame()
    result["data"], _ = pd.factorize(result["data"])
    return result.to_dict("list")["data"]


if __name__ == "__main__":
    # all_enterprise = query_all_enterprise()
    # print(all_enterprise)
    # print(len(all_enterprise))
    attr = '公司类型'
    t = F.one_hot(torch.tensor(get_enterprise_data(attr)))
    print(t)
    print(t.shape)
