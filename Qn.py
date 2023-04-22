import numpy as np
def Qn(ln,F,n,N):
        Q1 = np.zeros(N+1)
        Q2 = np.zeros(N+1)
        Q3 = np.zeros(N+1)
        # Q_1
        for m in np.arange(0, n + 1):
            Q1 += F[n - m] * ln[m]
        # Q_2
        for m in np.arange(n + 1, N + 1):
            Q2 += F[m - n] * ln[m]
        # Q_3
        for m in np.arange(1, N - n + 1):
            Q3 += F[n + m] * ln[m]
        qn = 0.5*(Q1 + Q2 + Q3)
        return qn