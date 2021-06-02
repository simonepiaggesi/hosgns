import networkx as nx
import numpy as np
import itertools as it
import scipy
import time
import sys
import pandas as pd
import os
import sparse

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import cartesian


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_random_walks_from_supra_allnodes_blockdiag(supra_H_und, window_size):
    
    '''
    Compute co-occurrence probabilities over random walks, passing into block diagonal matrix, 
    returning all couples (w_t, c_s).
    
    Parameters
    ----------
    supra_H_und : undirected networkx supra-adjacency graph
    window_size : contexts widonw over random walks
    
    Returns
    -------
    Pij : scipy sparse matrix of dimensions |V||T| x |V||T|
    '''
    
    node_name = list(supra_H_und.nodes())
    NR_NODES = np.unique([int(n_node.split('-')[0]) for n_node in node_name]).shape[0]
    NR_TIMES = np.unique([int(n_node.split('-')[1]) for n_node in node_name]).shape[0]

    components_nodelist = [list(supra_H_und.subgraph(n0).nodes()) for n0 in nx.connected_components(supra_H_und)]
    components_nodelist = [n0 for l in components_nodelist for n0 in l]
    
    G = nx.to_scipy_sparse_matrix(supra_H_und, nodelist=components_nodelist, format='csr', dtype=np.float32)
    volG = scipy.sparse.csr_matrix.sum(G)
    dout = np.array(list(dict(supra_H_und.degree(weight='weight', nbunch=components_nodelist)).values()))

    invD = scipy.sparse.diags(diagonals=(dout+1e-16)**(-1), format='csr', dtype=np.float32) 
    P = invD @ G

    D = scipy.sparse.diags(diagonals=dout, format='csr', dtype=np.float32) 
    Pij = D @ P + P.T @ D
    if window_size > 1:
        Pr = P.copy()
        for r in range(window_size-1):
            Pr = Pr @ P
            Pij += D @ Pr + Pr.T @ D
    Pij /= 2 * window_size * volG
    
    
    ordered_nodelist = list(it.product(range(NR_NODES), range(NR_TIMES)))
    df_ordered_nodelist = pd.DataFrame(ordered_nodelist, columns = ['i', 'tslice']).astype(int)
    df_components_nodelist = pd.DataFrame([n0.split('-') for n0 in components_nodelist], columns = ['i', 'tslice']).reset_index().astype(int)
    
    return_index = df_ordered_nodelist.merge(df_components_nodelist,\
                            on=['i', 'tslice'], how='left').loc[:,'index'].values
    
    return Pij[np.ix_(return_index, return_index)]#Pij[return_index_row, return_index_col]

def make_random_walks_from_supra_allnodes(supra_H_und, window_size):
    '''
    Compute co-occurrence probabilities over random walks, returning all couples (w_t, c_s).
    
    Parameters
    ----------
    supra_H_und : undirected networkx supra-adjacency graph
    window_size : contexts widonw over random walks
    
    Returns
    -------
    Pij : scipy sparse matrix of dimensions |V||T| x |V||T|
    '''
    
    node_name = list(supra_H_und.nodes())
    NR_NODES = np.unique([int(n_node.split('-')[0]) for n_node in node_name]).shape[0]
    NR_TIMES = np.unique([int(n_node.split('-')[1]) for n_node in node_name]).shape[0]

    ordered_nodelist = list(map(lambda x: str(x[0])+'-'+str(x[1]), it.product(range(NR_NODES), range(NR_TIMES))))
    
    G = nx.to_scipy_sparse_matrix(supra_H_und, nodelist=ordered_nodelist, format='csr', dtype=np.float32)
    volG = scipy.sparse.csr_matrix.sum(G)
    dout = np.array(list(dict(supra_H_und.degree(weight='weight', nbunch=ordered_nodelist)).values()))

    invD = scipy.sparse.diags(diagonals=(dout+1e-16)**(-1), format='csr', dtype=np.float32) 
    P = invD @ G

    D = scipy.sparse.diags(diagonals=dout, format='csr', dtype=np.float32) 
    Pij = D @ P + P.T @ D
    if window_size > 1:
        Pr = P.copy()
        for r in range(window_size-1):
            Pr = Pr @ P
            Pij += D @ Pr + Pr.T @ D
    Pij /= 2 * window_size * volG
    
    return Pij

def make_random_walks_from_supra(supra_G_und, window_size, nodelist=None):
    '''
    Compute co-occurrence probabilities over random walks, returning only couples (w_t, c_s) of active nodes.
    
    Parameters
    ----------
    supra_H_und : undirected networkx supra-adjacency graph
    window_size : contexts widonw over random walks
    nodelist : node ordering based on user preference (if None, the default ordering is used)
    
    Returns
    -------
    Pij : scipy sparse matrix of dimensions |V^(T)| x |V^(T)|
    '''
    
    if nodelist==None:
        nodelist = supra_G_und.nodes()   
    
    G = nx.to_scipy_sparse_matrix(supra_G_und, nodelist=nodelist, format='csr', dtype=np.float32)
    volG = scipy.sparse.csr_matrix.sum(G)
    dout = np.array(list(dict(supra_G_und.degree(weight='weight', nbunch=nodelist)).values()))

    invD = scipy.sparse.diags(diagonals=1/dout, format='csr', dtype=np.float32)
    P = invD @ G
    
    D = scipy.sparse.diags(diagonals=dout, format='csr', dtype=np.float32) 
    
    Pij = D @ P + P.T @ D
    if window_size > 1:
        Pr = P.copy()
        for r in range(window_size-1):
            Pr = Pr @ P
            Pij += D @ Pr + Pr.T @ D
    Pij /= 2 * window_size * volG
    
    return Pij
    
def reshape_supra_to_3way_sparse(supra_H, Pij, reduce_contexts=False):
    '''
    Reshape a scipy sparse matrix |V||T| x |V||T| in a 3way sparse pydata tensor.
    
    Parameters
    ----------
    supra_H : networkx supra-adjacency graph
    Pij : scipy sparse matrix of dimensions |V||T| x |V||T|
    reduce_contexts : to reduce the second axis of dimension |V||T| to |V| (summing over the second time dimension)
    
    Returns
    -------
    Pijk : 3way sparse pydata tensor of dimensions |V|x|T|x|V| (reduce_contexts=True) or |V|x|T|x|V||T| (reduce_contexts=False)
    '''
    
    node_name = list(supra_H.nodes())
    NR_NODES = np.unique([int(n_node.split('-')[0]) for n_node in node_name]).shape[0]
    NR_TIMES = np.unique([int(n_node.split('-')[1]) for n_node in node_name]).shape[0]
    Pijkl_sp = sparse.COO.from_scipy_sparse(Pij)
            
    if reduce_contexts:
        return Pijkl_sp.reshape((NR_NODES, NR_TIMES, NR_NODES, NR_TIMES))\
    .sum(axis=-1)
    else:
        return Pijkl_sp.reshape((NR_NODES, NR_TIMES, NR_NODES*NR_TIMES))


def reshape_supra_to_4way_sparse(supra_H, Pij):
    '''
    Reshape a scipy sparse matrix |V||T| x |V||T| in a 4way sparse pydata tensor.
    
    Parameters
    ----------
    supra_H : networkx supra-adjacency graph
    Pij : scipy sparse matrix of dimensions |V||T| x |V||T|
    
    Returns
    -------
    Pijkl : 4way sparse pydata tensor of dimensions |V|x|T|x|V|x|T|
    '''
    
    node_name = list(supra_H.nodes())
    NR_NODES = np.unique([int(n_node.split('-')[0]) for n_node in node_name]).shape[0]
    NR_TIMES = np.unique([int(n_node.split('-')[1]) for n_node in node_name]).shape[0]
    Pijkl_sp = sparse.COO.from_scipy_sparse(Pij)
    
    return Pijkl_sp.reshape((NR_NODES,NR_TIMES,NR_NODES,NR_TIMES))

def load_temp_data(dataset, aggr_time=10*60):
    '''
    Preprocess raw temporal network data.
    
    Parameters
    ----------
    dataset : ['LyonSchool', 'SFHH', 'LH10', 'InVS15', 'Thiers13']
    aggr_time : time window scale (default 600 seconds)
    
    Returns
    -------
    partial_times : sorted list of time slices
    s_temp_net : pandas series of events (i,j,tslice,weight) 
    df_tnet : pandas dataframe of events (i,j,tslice,weight) 
    '''
    
    df_temp_net = pd.read_csv(('../data/Data_SocioPatterns_20s_nonights/tij_%s.dat_nonights.dat' % dataset),
                        sep = '\t', header = None,
                        names = ['t', 'i', 'j'])
    # compute slice each contact event belongs to
    df_temp_net.loc[:,'tslice'] = np.floor((df_temp_net.t - df_temp_net.t.iloc[0]) / aggr_time)
    # group over (slice, i, j), and compute number of contacts within time slice,
    # regarded as "weight" for contacts in each time slice
    
    df_temp_net = df_temp_net[df_temp_net.i!=df_temp_net.j]
    s = df_temp_net['i'] > df_temp_net['j']
    df_temp_net.loc[s, ['i','j']] = df_temp_net.loc[s, ['j','i']].values
    df_temp_net.drop_duplicates(['t','i','j'], inplace=True)
    
    s_temp_net = df_temp_net.groupby(['tslice','i','j']).size().rename('weight')
    
    # times for all temporal slices, note that it may have a big gap (return to home)
    partial_times = sorted(list(s_temp_net.index.levels[0]))

    # convenience: dataframe version of the series above
    df_tnet = s_temp_net.reset_index()
    
    return partial_times, s_temp_net, df_tnet

def load_modified_temp_data(dataset, aggr_time=10*60):
    '''
    Preprocess partial temporal network data for link prediction.
    
    Parameters
    ----------
    dataset : ['LyonSchool', 'SFHH', 'LH10', 'InVS15', 'Thiers13']
    aggr_time : time window scale (default 600 seconds)
    
    Returns
    -------
    partial_times : sorted list of time slices
    s_temp_net : pandas series of events (i,j,tslice,weight) 
    df_tnet : pandas dataframe of events (i,j,tslice,weight) 
    '''
    
    df_temp_net = pd.read_csv(('../preprocessed/RemovedLinksTempNet/%s/tij_%s_7030_0.csv.gz' % (dataset,  dataset)),
                        sep = ',', header = None,
                        names = ['t', 'i', 'j'])
    # compute slice each contact event belongs to
    df_temp_net.loc[:,'tslice'] = np.floor((df_temp_net.t - df_temp_net.t.iloc[0]) / aggr_time)
    # group over (slice, i, j), and compute number of contacts within time slice,
    # regarded as "weight" for contacts in each time slice
    
    df_temp_net = df_temp_net[df_temp_net.i!=df_temp_net.j]
    s = df_temp_net['i'] > df_temp_net['j']
    df_temp_net.loc[s, ['i','j']] = df_temp_net.loc[s, ['j','i']].values
    df_temp_net.drop_duplicates(['t','i','j'], inplace=True)
    
    s_temp_net = df_temp_net.groupby(['tslice','i','j']).size().rename('weight')
    
    # times for all temporal slices, note that it may have a big gap (return to home)
    partial_times = sorted(list(s_temp_net.index.levels[0]))

    # convenience: dataframe version of the series above
    df_tnet = s_temp_net.reset_index()
    
    return partial_times, s_temp_net, df_tnet

#NODE CLASSIFICATION

def get_labels(dataset, pat_active_time):
    '''
    Return metadata labels for individual nodes.
    
    Parameters
    ----------
    dataset : ['LyonSchool', 'SFHH', 'LH10', 'InVS15', 'Thiers13']
    pat_active_time : list of 'node-time' strings 
    
    Returns
    -------
    list of class labels
    '''
    
    metadata_name = '../data/metadata/metadata_%s.dat' % dataset
    metadata = pd.read_csv(metadata_name, sep = '\t', header = None, index_col=0, names = ['label'])
    map_label = {label:i for i,label in enumerate(metadata.label.unique())}
    metadata.loc[:, 'label'] = metadata.label.map(map_label)
    return metadata.loc[np.unique([int(x.split('-')[0]) for x in pat_active_time])].label.values

def get_time_labels(dataset, pat_active_time):
    '''
    Return metadata labels for node-time tuples.
    
    Parameters
    ----------
    dataset : ['LyonSchool', 'SFHH', 'LH10', 'InVS15', 'Thiers13']
    pat_active_time : list of 'node-time' strings 
    
    Returns
    -------
    list of class labels
    '''
    
    metadata_name = '../data/metadata/metadata_%s.dat' % dataset
    metadata = pd.read_csv(metadata_name, sep = '\t', header = None, index_col=0, names = ['label'])
    map_label = {label:i for i,label in enumerate(metadata.label.unique())}
    metadata.loc[:, 'label'] = metadata.label.map(map_label)
    return metadata.loc[[int(x.split('-')[0]) for x in pat_active_time]].label.values

def get_infection_label(diff_model, beta, mu, pat_active_time, dataset, aggr_time=10*60, irun=1):
    '''
    Return S-I-R labels for active node-time tuples.

    Parameters
    ----------
    diff_model : 'SIR' 
    beta : infectious parameter
    mu : recovery parameter
    pat_active_time : list of 'node-time' strings 
    dataset : ['LyonSchool', 'SFHH', 'LH10', 'InVS15', 'Thiers13']
    aggr_time : time window scale (default 600 seconds)
    irun : index of SIR run
    
    Returns
    -------
    list of SIR labels
    '''
    
    def compute_index(row):
        node_id = '%d-%d' % (row['node'], row['tslice'])
        try:
            return node_id
        except ValueError:
            return None
    
    file_name = '../data/spreading/%s_%s_beta%s_mu%s_run%s_%s.csv.gz' % \
                (diff_model, dataset, '{:.4f}'.format(beta), '{:.4f}'.format(mu), irun, aggr_time)
    exists = os.path.isfile(file_name)
    if exists:  
        infection_data = file_name # open(file_name,'r')
        inf_labels = pd.read_csv(infection_data, sep = '\t', header = None, names = ['node', 'tslice', 't', 'state'])
        #inf_labels.loc[:,'tslice'] = np.floor((inf_labels.t - inf_labels.t.iloc[0]) / aggr_time)
        #inf_labels2 = inf_labels.sort_values(by = 't').groupby(['node', 'tslice']).last().reset_index()
        inf_labels.loc[:,'vec_idx'] = inf_labels.apply(compute_index, axis = 1)
        inf_labels.dropna(inplace = True)
        inf_labels.set_index('vec_idx', inplace = True)
        Y_state = inf_labels.loc[pat_active_time].state.values
        return Y_state
    else:
        return None
    
def train_test_split_predict(X, y, n_splits, starting_test_size, node_active_list, random_state):
    
    '''
    Perform node classification with node-time splits.
    
    Parameters
    ----------
    X : numpy array |V(T)|x d of node-time embeddings 
    y : numpy array |V(T)|x d of node-time labels
    n_splits : number of train-test splits
    starting_test_size : fraction of nodes (or times) used for test sets
    node_active_list : list of tuples (node, time) with the same ordering as X and y
    random_state : random seed for splitting
    
    Returns
    -------
    list of -n_splits- dictionaries containing results of logistic regressions
    '''
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    
    nodes_idx = np.unique([n for n,t in node_active_list])
    times_idx = np.unique([t for n,t in node_active_list])
    
    df_active = pd.DataFrame(node_active_list, columns=['i', 'tslice'])
    df_active.reset_index(inplace=True)
    
    results_list = []              
    for s in range(n_splits):
                          
        nodes_train, nodes_test = train_test_split(nodes_idx, test_size=starting_test_size, random_state=random_state)
        times_train, times_test = train_test_split(times_idx, test_size=starting_test_size, random_state=random_state)
        
        train_df = pd.DataFrame(cartesian((nodes_train, times_train)), columns=['i', 'tslice'])
        test_df = pd.DataFrame(cartesian((nodes_test, times_test)), columns=['i', 'tslice'])
        
        embs_train_idx = df_active.merge(train_df, on=['i', 'tslice']).loc[:,'index'].values
        embs_test_idx = df_active.merge(test_df, on=['i', 'tslice']).loc[:,'index'].values

        model_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, n_jobs=6, random_state=random_state))
        y_test_pred = model_clf.fit(X[embs_train_idx], y[embs_train_idx]).predict(X[embs_test_idx])
                          
        test_result = {'f1_macro': f1_score(y[embs_test_idx], y_test_pred, average='macro'), \
                       'f1_micro': f1_score(y[embs_test_idx], y_test_pred, average='micro')}
        best_results = {'train_index': embs_train_idx,'test_index': embs_test_idx,\
                        'Y_test': y[embs_test_idx], 'Y_pred': y_test_pred, 'test_result': test_result}
        results_list.append(best_results)
                          
    return results_list

def make_train_test_splits_NC(n_splits, starting_test_size, node_active_list, random_state):
    
    '''
    Build node-time splits for node classification.
    
    Parameters
    ----------
    n_splits : number of train-test splits
    starting_test_size : fraction of nodes (or times) used for test sets
    node_active_list : list of tuples (node, time) with the same ordering as X and y
    random_state : random seed for splitting
    
    Returns
    -------
    list of -n_splits- dictionaries containing train-test indices for embedding vectors
    '''
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    
    nodes_idx = np.unique([n for n,t in node_active_list])
    times_idx = np.unique([t for n,t in node_active_list])
    
    df_active = pd.DataFrame(node_active_list, columns=['i', 'tslice'])
    df_active.reset_index(inplace=True)
    
    splits_list = []              
    for s in range(n_splits):
                          
        nodes_train, nodes_test = train_test_split(nodes_idx, test_size=starting_test_size, random_state=random_state)
        times_train, times_test = train_test_split(times_idx, test_size=starting_test_size, random_state=random_state)
        
        train_df = pd.DataFrame(cartesian((nodes_train, times_train)), columns=['i', 'tslice'])
        test_df = pd.DataFrame(cartesian((nodes_test, times_test)), columns=['i', 'tslice'])
        
        train_df = df_active.merge(train_df, on=['i', 'tslice'])#.loc[:,'index'].values
        test_df = df_active.merge(test_df, on=['i', 'tslice'])#.loc[:,'index'].values
        
        emb1_train_idx = train_df.i.values
        emb2_train_idx = train_df.tslice.values
        y_train_idx = train_df.loc[:,'index'].values
        
        emb1_test_idx = test_df.i.values
        emb2_test_idx = test_df.tslice.values
        y_test_idx = test_df.loc[:,'index'].values
        
        splits_list.append({'train':(emb1_train_idx, emb2_train_idx, y_train_idx),
                            'test': (emb1_test_idx, emb2_test_idx, y_test_idx)})
                          
    return splits_list


#LINK RECONSTRUCTION

def build_dataset_LR(nodes_,contexts_,times_, df_events, df_active, random_state):
    
    '''
    Build a balanced set of positive-negative events (i,j,k) randomly.
    
    Parameters
    ----------
    nodes_ : set of indices i
    contexts_ : set of indices j
    times_ : set of indices k
    df_events : pandas dataframe of events (i,j,tslice,weight) 
    df_active : pandas dataframe of active noded (i,tslice)
    random_state : random seed for negative events
    
    Returns
    -------
    dataframe of events (i,j,tslice,label) 
    '''
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    
    #all possible combinations
    df_ = pd.DataFrame(cartesian((nodes_, contexts_, times_)), columns=['i', 'j', 'tslice'])
    
    #remove self-loops and sort ij along rows 
    df_ = df_[df_.i!=df_.j]
    s = df_['i'] > df_['j']
    df_.loc[s, ['i','j']] = df_.loc[s, ['j','i']].values
    df_.drop_duplicates(['i','j','tslice'], inplace=True)
    
    #remove links with inactive nodes
    df_ = df_.merge(df_active, on=['i', 'tslice'], suffixes=('', 'x'))\
             .merge(df_active, on=['j', 'tslice'], suffixes=('', 'x'))
    df_ = df_.loc[:, ['i','j','tslice']]
    
    #add labels
    df_ = df_.merge(df_events.loc[:, ['i', 'j', 'tslice', 'weight']], on=['i', 'j', 'tslice'], how='left')
    df_.loc[:, 'label'] = ~df_.weight.isnull()
    del df_['weight']
    
    #make balanced labels
    df_ = pd.concat([df_[df_.label], df_[~df_.label].sample(n=df_.label.sum(),random_state=random_state)])\
                .sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df_.astype(np.int32)

def make_train_test_splits_LR(n_splits, starting_test_size, node_active_list, df_events, random_state):
    
    '''
    Make node-node-time splits for link reconstruction.
    
    Parameters
    ----------
    n_splits : number of train-test splits
    starting_test_size : fraction of nodes (or times) used for test sets
    node_active_list : list of tuples (node, time)
    df_events : pandas dataframe of events (i,j,tslice,weight)
    random_state : random seed for splitting
    
    Returns
    -------
    list of -n_splits- dictionaries containing train-test indices for embedding vectors
    '''
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
        
    nodes_idx = np.unique([n for n,t in node_active_list])
    times_idx = np.unique([t for n,t in node_active_list])
    
    df_active = pd.DataFrame(node_active_list, columns=['i', 'tslice'])
    df_active.loc[:, 'j'] =df_active.loc[:, 'i'] 
    df_active.reset_index(inplace=True)
    
    splits_list = []              
    for s in range(n_splits):
             
        nodes_train, nodes_test = train_test_split(nodes_idx, test_size=starting_test_size, random_state=random_state)
        times_train, times_test = train_test_split(times_idx, test_size=starting_test_size, random_state=random_state)
        
        train_df = build_dataset_LR(nodes_train, nodes_train, times_train, \
                                 df_events, df_active.loc[:, ['i','j','tslice']], random_state)
        test_df = build_dataset_LR(nodes_test, nodes_test, times_test, \
                                 df_events, df_active.loc[:, ['i','j','tslice']], random_state)
        
        y_train = train_df.label.values
        y_test = test_df.label.values
        
        emb1_train_idx = train_df.i.values
        emb2_train_idx = train_df.j.values
        emb3_train_idx = train_df.tslice.values
        
        emb1_test_idx = test_df.i.values
        emb2_test_idx = test_df.j.values
        emb3_test_idx = test_df.tslice.values
        
        splits_list.append({'train':(emb1_train_idx, emb2_train_idx, emb3_train_idx, y_train),
                            'test': (emb1_test_idx, emb2_test_idx, emb3_test_idx, y_test)})
                          
    return splits_list

#LINK PREDICTION

def build_dataset_train_LP(nodes_, contexts_, times_, df_events, df_active, random_state):
    
    '''
    Build a balanced set of positive-negative events (i,j,k) randomly.
    
    Parameters
    ----------
    nodes_ : set of indices i
    contexts_ : set of indices j
    times_ : set of indices k
    df_events : pandas dataframe of events (i,j,tslice,weight) 
    df_active : pandas dataframe of active noded (i,tslice)
    random_state : random seed for negative events
    
    Returns
    -------
    dataframe of events (i,j,tslice,label) 
    '''
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
        
    df_links, df_drop = df_events 
    
    #all possible combinations
    df_ = pd.DataFrame(cartesian((nodes_, contexts_, times_)), columns=['i', 'j', 'tslice'])
    
    #remove self-loops and sort ij along rows 
    df_ = df_[df_.i!=df_.j]
    s = df_['i'] > df_['j']
    df_.loc[s, ['i','j']] = df_.loc[s, ['j','i']].values
    df_.drop_duplicates(['i','j','tslice'], inplace=True)
    
    #remove links with inactive nodes
    df_ = df_.merge(df_active, on=['i', 'tslice'], suffixes=('', 'x'))\
             .merge(df_active, on=['j', 'tslice'], suffixes=('', 'x'))
    df_ = df_.loc[:, ['i','j','tslice']]
    
    #remove dropped links
    df_ = df_.merge(df_drop, on=['i', 'j', 'tslice'], how='left').loc[lambda df: df.weight.isnull()]
    del df_['weight']
    
    #add labels
    df_ = df_.merge(df_links.loc[:, ['i', 'j', 'tslice', 'weight']], on=['i', 'j', 'tslice'], how='left')
    df_.loc[:, 'label'] = ~df_.weight.isnull()
    del df_['weight']
    
    #make balanced labels
    df_ = pd.concat([df_[df_.label], df_[~df_.label].sample(n=df_.label.sum(), random_state=random_state)])\
                .sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df_.astype(np.int32)


def build_dataset_test_LP(nodes_,contexts_,times_, df_events, df_active, random_state):
    
    '''
    Build a balanced set of positive-negative events (i,j,k) randomly.
    
    Parameters
    ----------
    nodes_ : set of indices i
    contexts_ : set of indices j
    times_ : set of indices k
    df_events : pandas dataframe of events (i,j,tslice,weight) 
    df_active : pandas dataframe of active noded (i,tslice)
    random_state : random seed for negative events
    
    Returns
    -------
    dataframe of events (i,j,tslice,label) 
    '''
    
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    
    df_links, df_drop = df_events 
    
    #all possible combinations
    df_ = pd.DataFrame(cartesian((nodes_, contexts_, times_)), columns=['i', 'j', 'tslice'])
    
    #remove self-loops and sort ij along rows 
    df_ = df_[df_.i!=df_.j]
    s = df_['i'] > df_['j']
    df_.loc[s, ['i','j']] = df_.loc[s, ['j','i']].values
    df_.drop_duplicates(['i','j','tslice'], inplace=True)
    
    #remove links with inactive nodes
    df_ = df_.merge(df_active, on=['i', 'tslice'], suffixes=('', 'x'))\
             .merge(df_active, on=['j', 'tslice'], suffixes=('', 'x'))
    df_ = df_.loc[:, ['i','j','tslice']]
    
    #remove active links
    df_ = df_.merge(df_links, on=['i', 'j', 'tslice'], how='left').loc[lambda df: df.weight.isnull()]
    del df_['weight']
    
    #add labels
    df_ = df_.merge(df_drop.loc[:, ['i', 'j', 'tslice', 'weight']], on=['i', 'j', 'tslice'], how='left')
    df_.loc[:, 'label'] = ~df_.weight.isnull()
    del df_['weight']
    
    #make balanced labels
    df_ = pd.concat([df_[df_.label], df_[~df_.label].sample(n=df_.label.sum(), random_state=random_state)])\
                .sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df_.astype(np.int32)

def make_train_test_splits_LP(n_splits, starting_test_size, node_active_list, df_events, random_state):
    
    '''
    Make node-node-time splits for link prediction.
    
    Parameters
    ----------
    n_splits : number of train-test splits
    starting_test_size : fraction of nodes (or times) used for test sets
    node_active_list : list of tuples (node, time)
    df_events : pandas dataframe of events (i,j,tslice,weight)
    random_state : random seed for splitting
    
    Returns
    -------
    list of -n_splits- dictionaries containing train-test indices for embedding vectors
    '''
    
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    
    nodes_idx = np.unique([n for n,t in node_active_list])
    times_idx = np.unique([t for n,t in node_active_list])
    
    df_active = pd.DataFrame(node_active_list, columns=['i', 'tslice'])
    df_active.loc[:, 'j'] = df_active.loc[:, 'i'] 
    df_active.reset_index(inplace=True)
    
    splits_list = []              
    for s in range(n_splits):
        
        nodes_train, nodes_test = train_test_split(nodes_idx, test_size=starting_test_size, random_state=random_state)
        times_train, times_test = train_test_split(times_idx, test_size=starting_test_size, random_state=random_state)
        
        train_df = build_dataset_train_LP(nodes_train, nodes_train, times_train, \
                                 df_events, df_active.loc[:, ['i','j','tslice']], random_state)
        test_df = build_dataset_test_LP(nodes_test, nodes_test, times_test, \
                                 df_events, df_active.loc[:, ['i','j','tslice']], random_state)
        
        y_train = train_df.label.values
        y_test = test_df.label.values
        
        emb1_train_idx = train_df.i.values
        emb2_train_idx = train_df.j.values
        emb3_train_idx = train_df.tslice.values
        
        emb1_test_idx = test_df.i.values
        emb2_test_idx = test_df.j.values
        emb3_test_idx = test_df.tslice.values
        
        splits_list.append({'train':(emb1_train_idx, emb2_train_idx, emb3_train_idx, y_train),
                            'test': (emb1_test_idx, emb2_test_idx, emb3_test_idx, y_test)})
                          
    return splits_list