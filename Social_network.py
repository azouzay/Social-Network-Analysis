#### Part 1 about Manipulating Graphs then Project 5

# # Creating and Manipulating Graphs
# 
# Eight employees at a small company were asked to choose 3 movies that they would most enjoy watching for the upcoming company movie night.
#These choices are stored in the file `Employee_Movie_Choices.txt`.
# 
# A second file, `Employee_Relationships.txt`, has data on the relationships between different coworkers. 
# 
# The relationship score has value of `-100` (Enemies) to `+100` (Best Friends). A value of zero means the two employees haven't interacted or are indifferent.
# 

import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms import bipartite


# Set of employees
employees = set(['Pablo',
                 'Lee',
                 'Georgia',
                 'Vincent',
                 'Andy',
                 'Frida',
                 'Joan',
                 'Claude'])

# Set of movies
movies = set(['The Shawshank Redemption',
              'Forrest Gump',
              'The Matrix',
              'Anaconda',
              'The Social Network',
              'The Godfather',
              'Monty Python and the Holy Grail',
              'Snakes on a Plane',
              'Kung Fu Panda',
              'The Dark Knight',
              'Mean Girls'])


# Loading the bipartite graph from `Employee_Movie_Choices.txt`     
Movie_Choices=nx.read_edgelist('Employee_Movie_Choices.txt', delimiter="\t",nodetype=str,encoding='utf-8')

# Add nodes attributes named `'type'` where movies have the value `'movie'` and employees have the value `'employee'` and return that graph.
Movie_Choices.add_nodes_from(movies, type='movie')
Movie_Choices.add_nodes_from(employees, type='employee')

# Find a weighted projection of the graph which tells us how many movies different pairs of employees have in common.
weighted_projection= bipartite.weighted_projected_graph(Movie_Choices, employees, ratio=False)
 
# Find the Pearson correlation between employee relationship scores and the number of movies they have in common.
#If two employees have no movies in common it should be treated as a 0, not a missing value, and should be included in the correlation calculation.
Employee_Relationships=nx.read_edgelist('Employee_Relationships.txt', delimiter="\t",data=[('relationship_score', int)]) 
Emp_rel=pd.DataFrame(Employee_Relationships.edges(data=True), columns=['From', 'To', 'relationship_score'])
movies=pd.DataFrame(weighted_projection.edges(data=True), columns=['From', 'To', 'movies'])
    
movies2 = movies.copy()
movies2.rename(columns={"From":"From1", "To":"From"}, inplace=True)
movies2.rename(columns={"From1":"To"}, inplace=True)
merge = pd.concat([movies, movies2])
merge = pd.merge(merge, Emp_rel, on = ['From', 'To'], how='right')
    
merge['relationship_score']=merge['relationship_score'].map(lambda x: x['relationship_score']if type(x)==dict else None)
merge['movies']=merge['movies'].map(lambda x: x['weight']if type(x)==dict else None)
    
merge.fillna(0, inplace=True)
    
Pearson_correlation= merge['relationship_score'].corr(merge['movies'])

#-----------

# ## Project 5 - Company Emails
# 
# For this project we will be working with a company's email network where each node corresponds to a person at the company,
#and each edge indicates that at least one email has been sent between two people.
# 
# The network also contains the node attributes `Department` and `ManagementSalary`.
# 
# `Department` indicates the department in the company which the person belongs to, and `ManagementSalary` indicates whether that person is receiving a management position salary.

import networkx as nx
import pandas as pd
import numpy as np
import pickle

G = nx.read_gpickle('email_prediction.txt')

print(nx.info(G))


#  Salary Prediction
# 
# Using network `G`, identify the people in the network with missing values for the node attribute `ManagementSalary` and predict whetheror not these individuals are receiving a management position salary.
# Predictions will need to be given as the probability that the corresponding employee is receiving a management position salary.
# 

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

def is_management(node):
    managementSalary = node[1]['ManagementSalary']
    if managementSalary == 0:
        return 0
    elif managementSalary == 1:
        return 1
    else:
        return None
        
df = pd.DataFrame(index=G.nodes())
df['clustering'] = pd.Series(nx.clustering(G))
df['degree'] = pd.Series(G.degree())
df['degree_centrality'] = pd.Series(nx.degree_centrality(G))
df['closeness'] = pd.Series(nx.closeness_centrality(G, normalized=True))
df['betweeness'] = pd.Series(nx.betweenness_centrality(G, normalized=True))
df['pr'] = pd.Series(nx.pagerank(G))
    
df['is_management'] = pd.Series([is_management(node) for node in G.nodes(data=True)])
    
df_train = df[~pd.isnull(df['is_management'])]
df_test = df[pd.isnull(df['is_management'])]
    
features = ['clustering', 'degree', 'degree_centrality', 'closeness', 'betweeness', 'pr']
X_train = df_train[features]
Y_train = df_train['is_management']
X_test = df_test[features]
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf = MLPClassifier(hidden_layer_sizes = [10, 5], alpha = 5,random_state = 0, solver='lbfgs', verbose=0)
clf.fit(X_train_scaled, Y_train)
test_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
salary_predictions= pd.Series(test_proba,X_test.index)

# Predict future connections between employees of the network. The future connections information has been loaded into the variable `future_connections`. The index is a tuple indicating a pair of nodes that currently do not have a connection, and the `Future Connection` column indicates if an edge between those two nodes will exist in the future, where a value of 1.0 indicates a future connection.

future_connections = pd.read_csv('Future_Connections.csv', index_col=0, converters={0: eval})
future_connections.head(10)

# Using network `G` and `future_connections`, identify the edges in `future_connections` with missing values and predict whether or not these edgeswill have a future connection.
# Your predictions will need to be given as the probability of the corresponding edge being a future connection.

for node in G.nodes():
    G.node[node]['community'] = G.node[node]['Department']
preferential_attachment = list(nx.preferential_attachment(G))
df = pd.DataFrame(index=[(x[0], x[1]) for x in preferential_attachment])
df['preferential_attachment'] = [x[2] for x in preferential_attachment]
    
cn_soundarajan_hopcroft = list(nx.cn_soundarajan_hopcroft(G))
df_cn_soundarajan_hopcroft = pd.DataFrame(index=[(x[0], x[1]) for x in cn_soundarajan_hopcroft])
df_cn_soundarajan_hopcroft['cn_soundarajan_hopcroft'] = [x[2] for x in cn_soundarajan_hopcroft]
df = df.join(df_cn_soundarajan_hopcroft,how='outer')
df['cn_soundarajan_hopcroft'] = df['cn_soundarajan_hopcroft'].fillna(value=0)
    
df['resource_allocation_index'] = [x[2] for x in list(nx.resource_allocation_index(G))]
    
df['jaccard_coefficient'] = [x[2] for x in list(nx.jaccard_coefficient(G))]
    
df = future_connections.join(df,how='outer')
df_train = df[~pd.isnull(df['Future Connection'])]
df_test = df[pd.isnull(df['Future Connection'])]
    
features = ['cn_soundarajan_hopcroft', 'preferential_attachment', 'resource_allocation_index', 'jaccard_coefficient']
X_train = df_train[features]
Y_train = df_train['Future Connection']
X_test = df_test[features]
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf = MLPClassifier(hidden_layer_sizes = [10, 5], alpha = 5,random_state = 0, solver='lbfgs', verbose=0)
                       
clf.fit(X_train_scaled, Y_train)
test_proba = clf.predict_proba(X_test_scaled)[:, 1]
predictions = pd.Series(test_proba,X_test.index)
target = future_connections[pd.isnull(future_connections['Future Connection'])]
target['prob'] = [predictions[x] for x in target.index]
    
Predict_future_connect= target['prob']

