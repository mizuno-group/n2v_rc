## sample data information
sample gene co-expression network data. 12436 genes and 31895 links are contained.

### items and contents
- sample_cor_gpickle : Graph object in Python pickle format
- sample_betweenness_centrality_dic.pkl : Node and its Betweenness Centrality dictionary saved in Python pickle format

### processing method
1. gene expression data was obtained from GSE4271 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE4271)
2. dataset was z-score normalized across the samples
3. calculated the relationship between genes using FDR-corrected correlation method and created relationship ranking
4. constructed network with planar filtered maximally graph (PMFG) from the relationship ranking
5. summarize the information using networkx and save each result
