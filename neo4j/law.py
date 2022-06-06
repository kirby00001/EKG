import torch
import pandas as pd

from neo4j import neo4j


def get_law_data(attr):
    result = neo4j.run(f"MATCH (n:LL_Entity) RETURN n.`{attr}` AS data")
    result = result.to_data_frame()
    result["data"], _ = pd.factorize(result["data"])
    return result.to_dict("list")["data"]


if __name__ == "__main__":
    print(torch.tensor(get_law_data('日期')))
    print(torch.tensor(get_law_data('案由')))
