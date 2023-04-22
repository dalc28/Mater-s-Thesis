import numpy as np
def D(z,N):
    D = np.zeros((N+1,N+1))
    c = np.ones(N+1)
    c[0] = 2
    c[N] = 2
    for j in np.arange(N+1):
        for k in np. arange(N+1):
            if j != k:
                D[j,k] = ((c[j])/(c[k]))*((-1)**(j+k))/(z[j] - z[k])
            else:
                if k == 0 & j == 0:
                    D[0,0] = ((2*(N**2))+1)/6
                else:
                    if k == N & j == N:
                        D[N,N] = -((2*(N**2))+1)/6
                    else:
                        D[j,k] = - (z[k])/(2*(1 - ((z[k]**2))))
    return D







