'''
Código elaborado para el analísis espacial de inestabilidades hidrodinámicas en un flujo cortante
mediante la implementación de métodos espectrales usando los polinomios de Chebyshev.
Autor: Diego Armando Landinez Capacho
Director: Guillermo Jaramillo pizarro
Univaersidad del valle
cali 2023
'''

#Análisis espacial de inestabilidades hidrodinámicas
##Software diseñado por Diego Armando Landinez Capacho
##Universidad de Valle, colombia,2023.
## Paquetes a usar
import numpy as np
import scipy.linalg as la
import numpy.polynomial.chebyshev as npcheby
import pandas as pd
import matplotlib.pyplot as plt
from D import D
from m_op1 import m_op1
#from m_op import m_op
N = 100
bethar = 0.2
#yinf = (2*np.pi)/alphar
#Aplicamos una transformación para mapear los valores de y a los de z que corresponden al dominio de los polinomios de
#Chebyshev
# r es el factor de escala de la transformación recomendado como 2.0 para el método de colocación.
r = 2.0
z = np.cos((np.arange(0,N+1)*np.pi)/N)
z[0] = 0.999999
z[N] = -0.999999
#m = ((1-(z**2))**(3/2))/r
#m2 = ((1-(z**2))**3)/(r**2)
#dm = ((-3*z)/r)*(np.sqrt(1-(z**2)))
#u = 0.5*(1 + np.tanh((z*r)/(np.sqrt(1-(z**2)))))
#du = ((r/(np.sqrt(1-(z**2))))*(1/((np.cosh((z*r)/(np.sqrt(1-(z**2)))))**2)))*(1+((z**2)/(1-(z**2))))
#du2 = ((r/(1-(z**2)))*(1+((z**2)/(1-(z**2))))*(1/((np.cosh((z*r)/(np.sqrt(1-(z**2)))))**2)))*(((3*z)/(np.sqrt(1-(z**2))))-((2*r)*(1+((z**2)/(1-(z**2))))*(np.tanh((z*r)/(np.sqrt(1-(z**2)))))))
## definimos las funcioines f
U = lambda z: 0.5*(1+np.tanh((z*r)/(np.sqrt(1-(z**2)))))
#du = lambda z: 0.5*((r/(np.sqrt(1-(z**2))))*(1/((np.cosh((z*r)/(np.sqrt(1-(z**2)))))**2)))*(1+((z**2)/(1-(z**2))))
#du2 = lambda z: 0.5*((r/(1-(z**2)))*(1+((z**2)/(1-(z**2))))*(1/((np.cosh((z*r)/(np.sqrt(1-(z**2)))))**2)))*(((3*z)/(np.sqrt(1-(z**2))))-((2*r)*(1+((z**2)/(1-(z**2))))*(np.tanh((z*r)/(np.sqrt(1-(z**2)))))))
#m = lambda z: ((1-(z**2))**(3/2))/r
#m2 = lambda z: ((1-(z**2))**3)/(r**2)
#mdm = lambda z: ((-3*z)/r**2)*((1-(z**2))**2)
f1 = lambda z: (0.5*(1+np.tanh((z*r)/(np.sqrt(1-(z**2))))))*(((1-(z**2))**3)/(r**2)) ##Um^2
f2 = lambda z: (0.5*(1+np.tanh((z*r)/(np.sqrt(1-(z**2))))))*(((-3*z)/r**2)*((1-(z**2))**2)) #Umdm
f3 = lambda z: -0.5*((r/(1-(z**2)))*(1+((z**2)/(1-(z**2))))*(1/((np.cosh((z*r)/(np.sqrt(1-(z**2)))))**2)))*(((3*z)/(np.sqrt(1-(z**2))))-((2*r)*(1+((z**2)/(1-(z**2))))*(np.tanh((z*r)/(np.sqrt(1-(z**2))))))) ## -m(mU')'
f4 = lambda z: -bethar*(((1-(z**2))**3)/(r**2)) #betha*(U^2)
f5 = lambda z: -bethar*(((-3*z)/r**2)*((1-(z**2))**2)) #betha*(mdm)
##############################GRAFICAS#########################################################################################
#plt.plot(z,f4(z))
#plt.plot(z,f5(z))
#plt.plot(z,f3(z))
#plt.grid()
#plt.show()
###############################################################################################################################

#Construcciones de la matrices f
u = np.identity(N+1)*U(z)
f1 = np.identity(N+1)*f1(z)
f2 = np.identity(N+1)*f2(z)
f3 = np.identity(N+1)*f3(z)
f4 = np.identity(N+1)*f4(z)
f5 = np.identity(N+1)*f5(z)

## Construcción de las matrices de diferenciacion
#D = np.zeros((N+1,N+1))
#print(z)
#print(np.shape(z))
D = D(z,N)
D2 = D@D

## Construción de los vectores con las condiciones de frontera
#Condiciones de frontera
# J_1: ϕ(y_inf) = 0
J_1 = np.ones(N+1)
# J_2: ϕ(-y_inf) = 0
J_2 = np.ones(N+1)
J_2[1::2] = -1
## Construcción de la matriz lambda C0x^3 + C1x^2 + C2x + C3
C0 = -u
#C0i = la.inv(-u)
C1 = bethar*np.identity(N+1)
C2 = f1@D2 + f2@D + f3
C3 = f4@D2 + f5@D
## incorporar las condiciones de frontera a las matrices

C3[N] = J_1
C3[N-1] = J_2
C0[N] = np.zeros(N+1)
C0[N-1] = np.zeros(N+1)
C1[N] = np.zeros(N+1)
C1[N-1] = np.zeros(N+1)
C2[N] = np.zeros(N+1)
C2[N-1] = np.zeros(N+1)

#C0i = la.inv(C0)
#print(C0)
## Recucción de orden del sistema matricial
## Operaciones de columnas para la matriz C3

C3 = m_op1(C3,N)
#print(C3)
C0 = C0[:N-1,:N-1]
#print(C3[N-1])
#C0i = la.inv(C0)
C1 = C1[:N-1,:N-1]
C2 = C2[:N-1,:N-1]
C3 = C3[:N-1,:N-1]

##Construcción de la matriz Lambda
I = np.identity(N-1)
ceros = np.zeros([N-1,N-1])
#A = np.block([[-C0i@C1,-C0i@C2,-C0i@C3],[I,ceros,ceros],[ceros,I,ceros]])
A = np.block([[-C1,-C2,-C3],[I,ceros,ceros],[ceros,I,ceros]])
B = np.block([[C0,ceros,ceros],[ceros,I,ceros],[ceros,ceros,I]])
#print(A)
#print(B)
## Resolución del problema de valores propios no generalizado
eigenvalues, eigenvectors = la.eig(A,B,check_finite=False)
eigenvalues = np.extract(eigenvalues != np.inf, eigenvalues)
## Filtrado de los valores propiosque cumplen con la condición de estabilidad
eigenv = eigenvalues
#eigenv = eigenvalues
#print(len(eigenv))
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
#print(Alpha[alphamax[0]])
print(eigenvalues)
#print(Alpha)
