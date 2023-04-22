import numpy as np
def Sn(P,u,n,N):
    s1 = 0
    s2 = 0
    s3 = 0
    for m in np.arange(n + 1):
        s1 += 0.5 * P[n - m]*u[m]
    for m in np.arange(n + 1, N + 1):
        s2 += 0.5 * P[m - n]*u[m]
    for m in np.arange(1, N - n + 1):
        s3 += 0.5 * P[n + m]*u[m]
    sn = s1 + s2 + s3
    return sn