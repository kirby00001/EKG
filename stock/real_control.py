import pandas as pd
from neo4j.stock import query_stock_structure
from neo4j.enterprise import query_all_enterprise


def insert_real_controller():
    """
    计算所有企业的实际控制人
    :return:none
    """
    # 查询所有企业的pid
    enterprises = query_all_enterprise()
    length = len(enterprises)
    result = {"pid": [], "name": []}
    for index, enterprise in enumerate(enterprises):
        print(f"{enterprise}:", end="")
        # 获取股权结构
        stock_structure = query_stock_structure(enterprise, [enterprise])
        # 计算实际控制人
        name, rate = calculate_real_controller(stock_structure, 1.0)
        result["pid"].append(enterprise)
        result["name"].append(name)
        print(f"{name}  {index}/{length}")
    return pd.DataFrame(result)


def calculate_real_controller(stock_structure, rate_sum):
    """
    根据股权结构计算实际控制人
    :param stock_structure: 某公司的股权结构
    :param rate_sum: 该公司的权重
    :return:该公司实际控制人的信息和权重
    """
    people = {"name": [], "rate": []}
    for entity in stock_structure:
        if '-' not in entity["rate"]:
            if entity["type"] == "people":
                # 对于人，计算其权重
                name = entity["name"]
                rate = rate_sum * float(entity["rate"].strip("%"))
            else:
                # 对于企业，计算其实际控制人及其权重
                name, rate = calculate_real_controller(entity["list"],
                                                       rate_sum * float(entity["rate"].strip("%")) / 100)
            people["name"].append(name)
            people["rate"].append(rate)
        else:
            continue
    # 排序，选择权重最高的人
    people = pd.DataFrame(data=people)
    people = people.sort_values(by=["rate"], ascending=False).reset_index(drop=True)
    if people.size > 0:
        return people.iloc[0, :].values
    else:
        return "null", 0


if __name__ == "__main__":
    # pid = "37900982392977"
    # print(calculate_real_controller(query_stock_structure(pid, [pid]), 1.0))
    controller = insert_real_controller()
    controller.to_csv("D:/PycharmProjects/EKG/controller.csv", index=False)
