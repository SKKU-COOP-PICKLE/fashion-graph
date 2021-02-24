import numpy as np
import scipy.sparse as sp

def expand_csr_adj(adj, count:int):
    """
    Expand csr adj matrix by adding empty adj rows and columns
    ex. add 2 in (4,4) -> (6,6) with emtpy adj rows and columns
    
    returns expanded adj csr matrix, expanded # of indexes 
    """
    r,c = adj.shape
    
    adj = sp.vstack(
        [adj, sp.csr_matrix(np.zeros((count, c)))])
    adj = sp.hstack(
        [adj, sp.csr_matrix(np.zeros((r, count)))])
    
    return adj, list(range(r,r+count))

