# sample data information

## processing method
1. gene expression data was obtained from GSE4271 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE4271)
2. dataset was z-score normalized across the samples
3. constructed network with planar filtered maximally graph (PMFG)
4. summarize the information using networkx and save each result