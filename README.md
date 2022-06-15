## 1. Application Scenarios
### 1.1 Key concepts
Incognito shareholder: In order to circumvent the law or for other reasons, borrow the name of others to set up a company or make capital contribution in the name of others, and in the company's articles of association, shareholder register and business registration, are recorded as the actual contributors of others.
### 1.2 Motivation
The hidden shareholders are the most frequent cases of legal disputes of the company. If we can identify the hidden shareholders of the company as early as possible, it will be of great help to the company investment, legal disputes and other problems.
## 2. Model
### 2.1 Data generation
The data of hidden shareholders are few and not targeted in the previous data collection. To address this issue, we generated some data that consisted mainly of shareholders and companies.
- The shareholders will be associated with a certain number of legal disputes in order to simulate the emergence of the hidden shareholders in the display due to legal issues.
- In addition, certain relationships will be generated between the hidden shareholders and the real shareholders, such as family and friends, in order to simulate the reality in which the revealed shareholders are more closely related to the hidden shareholders.
### 2.2 Linkage prediction model
We treat the real edge as a positive sample and generate a negative sample based on random noise to convert into a binary classification problem, for which an encoder+decoder model can be fitted.
#### 2.2.1 Data preprocessing
In the preprocessing part, we first use the Index coding method to encode the node features of the map, and then add reverse edges to the map to enhance the mobility of the information on the map, and finally perform negative sampling to get the negative sample data, and the current negative sampling ratio is 1:1.
#### 2.2.2 Encoder
Then the positive sample and negative sample data are input into the encoder, which uses GNN-based graph embedding, consisting of two layers of RGCN, which can utilize the information of neighboring entities within two hops in the graph embedding process.
#### 2.2.3 Decoder
The decoder uses the DistMult scoring function to take the embedding representation of a node as input and outputs the probability that the relationship between two nodes is an implicit shareholder relationship.
### 2.3 Model evaluation
We use AUC as the evaluation metric, and the optimal AUC is 89.2%
### 2.3 Prediction and result preservation
We also generate a small portion of test data for prediction and save the prediction results as CSV files for presentation on the front-end
## 3. Highlights
We proposed a solution to the problem of hidden shareholder identification, generated data based on realistic situations, designed a model, and got okay results. And, compared with existing enterprise knowledge graph products, they do not have the function of hidden shareholder discovery. In this regard, it is superior to the existing enterprise knowledge graph products.