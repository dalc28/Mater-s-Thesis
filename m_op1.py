import numpy as np

def m_op1(C3, N):
    k=0
    for n in np.arange(2):
        if C3[N-n, N-n] != 1:
            a = C3[N-n, N-n]
            C3[:,N-n] = (1/a)*C3[:,N-n]
        for m in np.arange(N-n):
            a1 = C3[N-n,N-1-m-k]
            if a1 != 0:
                if a1 != 1:
                    C3[:,N-1-m-k] = (1/a1)*C3[:,N-1-m-k]
                C3[:,N-1-m-k] = C3[:,N-1-m-k] - C3[:,N-n]
            else:
                continue
        k += 1
    return C3