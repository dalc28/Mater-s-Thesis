import numpy as np
def Dn(n,N):
    D = np.zeros(N+1)
    if n == 0:
        for p in np.arange(n,N+1):
            if p % 2 == 1:
                D[p] = p
    else:
        for p in np.arange(n,N+1):
            if (p + n) % 2 == 1:
                D[p] = 2*p
    return D