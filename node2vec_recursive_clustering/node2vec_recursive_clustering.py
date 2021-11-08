# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:05:58 2021

1. Transform network structure to d-dimensional vectors using node2vec
2. Repeat k-means clustering recursively on the embedded vectors according to the following update conditions
    (i) Increase the k-split number, and consider the point where Newman modularity (Q) cannot be updated ns (default=10) times in a row as the optimal number of partitions.
    (ii) After the first time (recursive step), the following conditions are added additionally for the partition optimized by Q shown in (i)
    
        - target parent module size > min_threshold (default=0.8)
        - child module rich ratio (size > min threshold) > rich_threshold (default=10)
        - child module modularity (Q) > min_q (default=0.3)
        - any child compactness (v = (SPD_V / log(|V|)^α)) is smaller than parent one (defalut:α=1.0)
        
        Repeat recursively until no more partitions can be made that satisfy these conditions.

@author: I.Azuma
"""
from vectorization import Vectorization
import copy
import math
import itertools
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.community as nx_comm
from sklearn.cluster import KMeans
from centrality_kmeans import Scratch_KMeans


class Node2Vec_Recursive_Clustering(Vectorization):
    def __init__(self):
        super(Node2Vec_Recursive_Clustering,self).__init__()
        self.__first_children = list
        self.__first_vl = list
        self.__v0 = None
        self.__final = list
        self.__module_df = None
    
    def vec_prepare(self,g,dimensions=128,walk_length=100,num_walks=20,p=1,q=1,workers=4,window=15,min_count=1,sg=1):
        """
        transform network structure to d-dimensional vectors using node2vec
        """
        Vectorization.setg(self,g) # set graph information
        Vectorization.set_parameter(self,dimensions=dimensions,walk_length=walk_length,num_walks=num_walks,p=p,q=q,workers=workers,window=window,min_count=min_count,sg=sg)
        Vectorization.conduct_vectorization(self)
        
    def set_existing(self,g,df):
        """
        you can set embedded d-dimensional vectors directly
        """
        self.__g = g
        self.__vec_df = df
    
    def first_step(self,alpha=1.0,spd=0,centrality_centroid=False,centrality_dic:dict={},do_plot=True,spd_consideration=False,start=2,ns=10):
        """
        Determine the optimal number of k-splits from the transition of Q
        """
        if spd == 0:
            spd = nx.average_shortest_path_length(self.__g) # average shortest path length of all node pairs
        else:
            pass
        print("whole network SPD :",spd)
        v0 = spd/(math.log(len(self.__g.nodes)))**alpha # parent compactness
        self.__v0 = v0
        if centrality_centroid:
            print('KMeans with centrality centroid')
            if len(centrality_dic)==0:
                print('calculate centrality from now on ! This process time consuming !!')
            else:
                print('use existing centrality dict')
        else:
            print('KMenas with default KMeans++')
        print("")
        # decide k-split with modularity (determine initial centroid with network centrality or default KMeans++)
        first_children, first_result, first_qmax = modularity_kmeans(self.__g,self.__vec_df,centrality_dic=centrality_dic,target=set(self.__g.nodes()),centrality_centroid=centrality_centroid,do_plot=do_plot,spd_consideration=spd_consideration,start=start,ns=ns) 
        
        # evaluate each module after K-Means
        first_vl = []
        for c in first_children:
            first_vl.append(calc_compactness(self.__g,c))
        w = 0
        for vl in first_vl:
            if vl < v0:
                w += 1 # the child is more compact than parent
            else:
                pass
        if w==0:
            print("split was denied")
        else:
            print("split was accepted")
            print(w*100/len(first_vl), '% modules are more compact than initial structure')
            
            # plot second child size distribution
            first_len = [len(x) for x in first_children]
            x = [i+2 for i in range(len(first_children))] # start from k=2
            plt.bar(x,first_len)
            plt.xticks(x)
            plt.ylabel('frequency')
            plt.show()
            
            self.__first_children = first_children
            self.__first_vl = first_vl
    
    def get_first_step(self):
        return (self.__v0,self.__first_children,self.__first_vl)
    
    def set_first_step(self,v0:float,first_children:list,first_vl:list):
        """
        set first step result
        - use to evaluate robustness of recursive_step with the same first step result
        """
        self.__v0 = v0
        self.__first_children = first_children
        self.__first_vl = first_vl
        
    def recursive_step(self,min_threshold:int=10,rich_threshold:float=0.8,min_q:float=0.3,do_plot=False,centrality_dic:dict={},centrality_centroid=False,ns:int=10):
        """
        update condition
            1. target parent module size > min_threshold
            2. child module rich ratio (size > min threshold) > rich_threshold
            3. child module modularity (Q) > min_q
            4. any child compactness (v = SPD_V / log(|V|)^α) is smaller than parent one

        Parameters
        ----------
        min_threshold : int
            the minimum size of target module. The default is 10.
        rich_threshold : float
            the module ratio which satisfy the minimum size constraint. The default is 0.8.
        min_q : float
            Newman modularity Q threshold. The default is 0.3.
        do_plot : bool
            Variable for whether to plot the change in modularity as the number of divisions increases. The default is False.
        centrality_dic : dict
            Dictionary for each gene and its centrality value (e.g. Betweenness Centrality, Degree Cetnrality). The default is {}.
        centrality_centroid : bool
            Variable for whether to consider network centrality to determine the initial centroid of k-means clustering.
            The default is False.
        ns : int, optional
            The point at which Q could not be updated for ns consecutive times is regarded as the optimal number of divisions. The default is 10.
        """
        target_module = copy.deepcopy(self.__first_children)
        target_vl = copy.deepcopy(self.__first_vl)
        
        self.__final = []
        while len(target_module)>0:
            t = target_module[0]
            vp = target_vl[0]
            # module size > min_threshold
            if len(t) > min_threshold:
                children, label, qmax = modularity_kmeans(self.__g,self.__vec_df,centrality_dic=centrality_dic,target=t,centrality_centroid=centrality_centroid,do_plot=do_plot,spd_consideration=False,ns=ns)
                
                size_list = [len(x) for x in children]
                if len(size_list)==0:
                    rich_ratio=0
                else:
                    rich_ratio = sum(i>min_threshold for i in size_list)/len(size_list)
                # meet the update demand
                if (rich_ratio >= rich_threshold) & (qmax >= min_q):
                    print('rich children ratio :',rich_ratio)
                    # calc children compactness
                    children_vl = []
                    for c in children:
                        children_vl.append(calc_compactness(self.__g,c))
                    w=0
                    for vl in children_vl:
                        if vl < vp:
                            w += 1
                        else:
                            pass
                    if w==0:
                        print("split was terminated (no compact children)")
                        print("")
                        target_module.pop(0)
                        target_vl.pop(0)
                        self.__final.append(t)
                    else:
                        print("split was accepted")
                        print("")
                        target_module.pop(0)
                        target_vl.pop(0)
                        target_module.extend(children)
                        target_vl.extend(children_vl)
                else:
                    print('rich children ratio :',rich_ratio)
                    print("split was terminated (rich ratio or Q problem)")
                    print("")
                    target_module.pop(0)
                    target_vl.pop(0)
                    self.__final.append(t)
            # moduls size ≦ min_threshold
            else:
                print("split was terminated (small size)")
                print("")
                target_module.pop(0)
                target_vl.pop(0)
                self.__final.append(t)
                
    def summarize_module(self):
        """
          V1     V2    V3    ・・・  Vn
        gene1  gene5  gene6     genex
        gene2   nan   gene7     nan
        gene3   nan   gene8     nan
        gene4   nan    nan      nan
        """
        self.__module_df = pd.DataFrame(self.__final,index=['module_'+str(i) for i in range(len(self.__final))]).T
        
    def get_module(self):
        return self.__module_df

# fxn
def modularity_kmeans(g,vec_df,centrality_dic:dict,target:set,centrality_centroid=False,do_plot=True,spd_consideration=False,start=2,ns=10):
    """
    main function of k-means clustering

    Parameters
    ----------
    g : TYPE
        DESCRIPTION.
    vec_df : DataFrame
        DESCRIPTION.
    centrality_dic : dict
        Dictionary for each gene and its centrality value (e.g. Betweenness Centrality, Degree Cetnrality).
    target : set
        target gene pair information.
    centrality_centroid : bool
        Variable for whether to consider network centrality to determine the initial centroid of k-means clustering. The default is False.
    do_plot : bool
        Variable for whether to plot the change in modularity as the number of divisions increases. The default is False.
    spd_consideration : bool
        Variable whether to consider the SPD between candidates to determine the centroid. The default is False.
    start : TYPE, optional
        Initial value for optimization with increasing number of divisions. The default is 2.
    ns : TYPE, optional
        The point at which Q could not be updated for ns consecutive times is regarded as the optimal number of divisions. The default is 10.

    Returns
    -------
    module_lsit : list
        [{gene set1},{gene set2},...{gene setk}]
    best_result : dataframe
        gene and its label at the best split
    qmax : float
        max modularity(Q) at the best split

    """
    sub_g = nx.Graph.subgraph(g,target) # extract subgraph from whole g
    module_vec = vec_df.loc[target] # extract module vectors from node2vec whole vectors
    
    q_res = []
    best_result = pd.DataFrame()
    best_k = 0
    qmax = 0
    counter = 0
    for i in range(start,len(target)):
        # perform K-Means
        k=i
        if centrality_centroid:
            clf = Scratch_KMeans(g,n_clusters=k,max_iter=300)
            clf.define_centrality(data=centrality_dic)
            clf.fit(module_vec,spd_consideration=spd_consideration)
            
            # summarize the clustering result
            result_labels = clf.labels_.tolist()
            result_df = pd.DataFrame({'module':result_labels},index=module_vec.index)
        else:
            clf = KMeans(n_clusters=k,n_init=100) # set n_init high to supress fluctuation
            result = clf.fit(module_vec)
            
            # summarize the clustering result
            result_labels = result.labels_.tolist()
            result_df = pd.DataFrame({'module':result_labels},index=module_vec.index)
        
        # process to apply networkx method
        module_list = []
        for j in range(k):
            module_list.append(set(result_df.groupby('module').get_group(j).index))
    
        # calc modularity (Q)
        q = nx_comm.modularity(sub_g,module_list)
        #print(q)
        if q > qmax:
            qmax = q
            best_result = result_df
            best_k = k
            counter = 0
        else:
            counter += 1
            if counter == ns: # cannot update ns times in a row
                break
            pass
        q_res.append(q)
        
        print(i,end=" ")
        
    print('')
    print('--- K-Means with Q ---')
    print("sub graph nodes :",len(sub_g.nodes()))
    print("sub graph edges :",len(sub_g.edges()))
    print('best split number :',best_k)
    print('its modularity (Q) :',qmax)
    #print('')
    
    # elements in each module
    module_list = []
    for i in range(best_k):
        module_list.append(set(best_result.groupby('module').get_group(i).index))
    
    if do_plot:
        plt.plot([i for i in range(start,start+len(q_res))],q_res)
        plt.xticks([i for i in range(start,start+len(q_res))])
        plt.show()
    #return best_k
    return module_list, best_result, qmax

        
def calc_compactness(g,target:set,alpha=1.0):
    """
    1. focus on a module which is detected as cluster with 'modularity KMeans'
    2. calculate the compactness about the module
    
    !! This process relys on the target size. You may need to random sampling. !!
    
    Parameters
    ----------
    target : set
        gene clusters after modularity-kmeans
    alpha : float, optional
        DESCRIPTION. The default is 1.0.

    Returns
    -------
    vl : float
        vl = SPD / (log(V.size))^alpha

    """
    if len(target)<2:
        vl = float('inf')
    else:
        sum_spd = 0
        for v in itertools.combinations(target, 2):
            s = nx.bidirectional_shortest_path(g, v[0], v[1])
            sum_spd += (len(s)-1)
        target_spd = sum_spd/len(list(itertools.combinations(target, 2))) # average shortest path length in the target module
        
        vl = target_spd/(math.log(len(target)))**alpha
    return vl
    

