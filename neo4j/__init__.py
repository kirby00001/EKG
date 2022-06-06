from py2neo import Graph

url = "bolt://10.194.162.89:7687"
neo4j = Graph(url, auth=("neo4j", "123"))
