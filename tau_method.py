'''
Código elaborado para el analísis espacial de inestabilidades hidrodinámicas en un flujo cortante
mediante la implementación de métodos espectrales usando los polinomios de Chebyshev.
Autor: Diego Armando Landinez Capacho
Director: Guillermo Jaramillo pizarro
Univaersidad del valle
cali 2023
'''

#Paquetes a implementar
import numpy as np
import scipy.linalg as la
import numpy.polynomial.chebyshev as npcheby
import matplotlib.pyplot as plt
import pandas as pd
from Dn import Dn
from Qn import Qn
from Rn import Rn
from Sn import Sn
from m_op1 import m_op1

# Numero de divisiones en el dominio
N = 100
# frecuencia de la inestabilidad
bethar = 0.2

#Aplicamos una transformación para mapear los valores de y a los de z que corresponden al dominio de los polinomios de
#Chebyshev
# r es el factor de escala de la transformación recomendado como 2.0 para el método de colocación.
r = 2
#Definimos las funcioines que se encuentran en la ecuación diferencial de Rayleigh
#Perfil de velocidad Tangente hiperbolico mapeado al dominio z [-1,1]
U = lambda z: 0.5*(1+np.tanh((z*r)/(np.sqrt(1-(z**2)))))
#du = lambda z: 0.5*((r/(np.sqrt(1-(z**2))))*(1/((np.cosh((z*r)/(np.sqrt(1-(z**2)))))**2)))*(1+((z**2)/(1-(z**2))))
#du2 = lambda z: ((r/(1-(z**2)))*(1+((z**2)/(1-(z**2))))*(1/((np.cosh((z*r)/(np.sqrt(1-(z**2)))))**2)))*(((3*z)/(np.sqrt(1-(z**2))))-((2*r)*(1+((z**2)/(1-(z**2))))*(np.tanh((z*r)/(np.sqrt(1-(z**2)))))))
# m  es definido como la métrica dz/dy
m = lambda z: ((1-(z**2))**(3/2))/r
# m al cuadrado
m2 = lambda z: ((1-(z**2))**3)/(r**2)
# derivada de la métrica
dm2_2 = lambda z: -((3*z)*((1-z**2)**2))/r**2
#dm = lambda z: (-3*z/(r**2))*((1-(z**2))**2)
#dm = lambda z: ((-3*z)/r)*(np.sqrt(1-(z**2)))
ui = npcheby.Chebyshev.interpolate(U,N,[-1,1])
u = ui.coef
#dui = npcheby.Chebyshev.interpolate(du,N,[-1,1])
#du = dui.coef
#du2i = npcheby.Chebyshev.interpolate(du2,N,[-1,1])
#du2 = du2i.coef
du = npcheby.chebder(u,1)
du = np.append(du,0)
du2 = npcheby.chebder(u,2)
du2 = np.append(du2,0)
du2 = np.append(du2,0)
mi = npcheby.Chebyshev.interpolate(m,N,[-1,1])
m = mi.coef
m2i = npcheby.Chebyshev.interpolate(m2,N,[-1,1])
m2 = m2i.coef
dm2_2i = npcheby.Chebyshev.interpolate(dm2_2,N,[-1,1])
dm2_2 = dm2_2i.coef
######################################################################################################################
#z = np.linspace(-1,1,50)
#plt.plot(z,mi(z))
#plt.plot(z,ui(z))
#plt.plot(z,dm2_2i(z))
#plt.grid()
#plt.show()
#print(z)
#print(zk)
#print(du2(zk))
#######################################################################################################################

## Construcción de las matrices NXN

## Construcción de las matrices f
f1 = np.zeros(N+1)
f2 = np.zeros(N+1)
f3 = np.zeros(N+1)
f4 = np.zeros(N+1)
f5 = np.zeros(N+1)

for n in np.arange(N+1):
    f1[n] = Sn(u,m2,n,N)
    f2[n] = Sn(u,dm2_2,n,N)
f3 = -du2
f4 = -bethar*m2
f5 = -bethar*dm2_2
## Determinación de las matrices

f1n = np.zeros((N+1,N+1))
f2n = np.zeros((N+1,N+1))
f3n = np.zeros((N+1,N+1))
f4n = np.zeros((N+1,N+1))
f5n = np.zeros((N+1,N+1))
rn = np.zeros((N+1,N+1))
sn = np.zeros((N+1,N+1))
ln = np.zeros((N+1,N+1))
D = np.zeros((N+1,N+1))

for n in np.arange(N+1):
    ##Matriz L_n
    D[n] = Dn(n,N)
    ##Matriz Rn
    rn[n] = Rn(u,n,N)
    ##Matriz F3
    f3n[n] = Rn(f3,n,N)

ln = D @ D


for n in np.arange(N+1):
    f1n[n] = Qn(ln,f1,n,N)
    f2n[n] = Qn(D,f2,n,N)
    f4n[n] = Qn(ln,f4,n,N)
    f5n[n] = Qn(D,f5,n,N)

## Construcción de las matrices de diferenciacion
D = np.zeros((N+1,N+1))
for n in np.arange(N):
    D[n] = Dn(n,N)
#D2 = D @ D
#print(np.shape(rn))

## Construción de los vectores con las condiciones de frontera
#Condiciones de frontera
# J_1: ϕ(y_inf) = 0
J_1 = np.ones(N+1)
# J_2: ϕ(-y_inf) = 0
J_2 = np.ones(N+1)
J_2[1::2] = -1
## Construcción de la matriz lambda C0x^3 + C1x^2 + C2x + C3
C0 = -rn
#C0i = la.inv(-u)
C1 = bethar*np.identity(N+1)
C2 = f1n + f2n + f3n
C3 = f4n + f5n
#print(C2)
## incorporar las condiciones de frontera a las matrices
#print(np.shape(C1))

C3[N] = J_1
C3[N-1] = J_2
C0[N] = np.zeros(N+1)
C0[N-1] = np.zeros(N+1)
C1[N] = np.zeros(N+1)
C1[N-1] = np.zeros(N+1)
C2[N] = np.zeros(N+1)
C2[N-1] = np.zeros(N+1)

#print(C3)
## Recucción de orden del sistema matricial
## Operaciones de columnas para la matriz C3
C3 = m_op1(C3,N)
#print(C3)

C0 = C0[:N-1,:N-1]
#print(C3[N-1])
C0i = la.inv(C0)
C1 = C1[:N-1,:N-1]
C2 = C2[:N-1,:N-1]
C3 = C3[:N-1,:N-1]
#print(np.shape(C3))

##Construcción de la matriz Lambda
I = np.identity(N-1)
ceros = np.zeros([N-1,N-1])
A = np.block([[-C0i@C1,-C0i@C2,-C0i@C3],[I,ceros,ceros],[ceros,I,ceros]])
#A = np.block([[-C1,-C2,-C3],[I,ceros,ceros],[ceros,I,ceros]])
B = np.block([[C0,ceros,ceros],[ceros,I,ceros],[ceros,ceros,I]])
#print(A)
#print(B)
## Resolución del problema de valores propios no generalizado
eigenvalues, eigenvectors = la.eig(A,B,check_finite=False)
eigenvalues = np.extract(eigenvalues != np.inf, eigenvalues)
## Filtrado de los valores propiosque cumplen con la condición de estabilidad
eigenv = eigenvalues
#eigenv = eigenvalues

eigenv1 = np.zeros(len(eigenv),dtype='complex_')
for i in np.arange(len(eigenv)):
    if np.real(eigenv[i])>0:
        if np.real(eigenv[i])<1:
            if np.imag(eigenv[i])<0:
                if np.imag(eigenv[i]) > -1:
                    eigenv1[i] = eigenv[i]

Alpha = np.extract(np.imag(eigenv1) != 0 , eigenv1)
#alphamax = np.max(np.imag(-Alpha))
#alphai = (-1)*np.imag(Alpha)
#alpha_val = np.where(alphai == np.amax(alphai))
alphamod = abs(Alpha)
alphamax = np.where(alphamod == np.amax(alphamod))
print(Alpha[alphamax[0]])
#print(eigenv)
print(Alpha)
