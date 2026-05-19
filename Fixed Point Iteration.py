import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
import time
from matplotlib.ticker import FormatStrFormatter
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from scipy.stats import t

def Ueq(rho, xi, um, rhom):
    return um*(xi/um-rho/rhom)

def GSOM(rho, xi, u, um, rhom, Nt, Nx, Nw, lamda):
    dt = 3/Nt
    dx = 1/Nx
    z = rho*xi
    for n in range(Nt-1):
        for i in range(Nx):
            if i == Nx-1:
                rho[n+1, i] = 1/2*(rho[n, 0]+rho[n, i-1])-dt/(2*dx)*(rho[n, 0]*u[n, 0]-rho[n, i-1]*u[n, i-1])
                z[n+1, i] = 1/2*(z[n, 0]+z[n, i-1])-dt/(2*dx)*(z[n, 0]*u[n, 0]-z[n, i-1]*u[n, i-1])+lamda*rho[n, i]*(Ueq(rho[n, i], xi[n, i], um, rhom)-u[n, i])
            else:
                rho[n+1, i] = 1/2*(rho[n, i+1]+rho[n, i-1])-dt/(2*dx)*(rho[n, i+1]*u[n, i+1]-rho[n, i-1]*u[n, i-1])
                z[n+1, i] = 1/2*(z[n, i+1]+z[n, i-1])-dt/(2*dx)*(z[n, i+1]*u[n, i+1]-z[n, i-1]*u[n, i-1])+lamda*rho[n, i]*(Ueq(rho[n, i], xi[n, i], um, rhom)-u[n, i])
        xi = z/rho
    return rho, xi

def HJB(C, u, rho, xi, Nt, Nx, Nw, lamda, sigma):
    dt = 3/Nt
    dx = 1/Nx
    dw = 1/Nw
    for n in range(Nt-2, -1, -1):
        for i in range(Nx):
            if i == Nx-1:
                for j in range(Nw):
                    if j == Nw-1:
                        Cx = (C[n+1, 0, j]-C[n+1, i-1, j])/(2*dx)
                        Cw = (C[n+1, i, 0]-C[n+1, i, j-1])/(2*dx)
                        Cww = (C[n+1, i, 0]-2*C[n+1, i, j]+C[n+1, i, j-1])/dx**2
                        C[n, i, j] = C[n+1, i, j]+dt*Ueq(rho[n+1, i], xi[n, i], um, rhom)*Cx-dt/2*(Cx-lamda*Cw)**2-2*dt*lamda*Ueq(rho[n+1, i], xi[n+1, i], um, rhom)*Cw+dt*sigma**2/2*(1-rho[n+1, i])*Cww
                    else:
                        Cx = (C[n+1, 0, j]-C[n+1, i-1, j])/(2*dx)
                        Cw = (C[n+1, i, j+1]-C[n+1, i, j-1])/(2*dx)
                        Cww = (C[n+1, i, j+1]-2*C[n+1, i, j]+C[n+1, i, j-1])/dx**2
                        C[n, i, j] = C[n+1, i, j]+dt*Ueq(rho[n+1, i], xi[n, i], um, rhom)*Cx-dt/2*(Cx-lamda*Cw)**2-2*dt*lamda*Ueq(rho[n+1, i], xi[n+1, i], um, rhom)*Cw+dt*sigma**2/2*(1-rho[n+1, i])*Cww
            else:
                for j in range(Nw):
                    if j == Nw-1:
                        Cx = (C[n+1, i+1, j]-C[n+1, i-1, j])/(2*dx)
                        Cw = (C[n+1, i, 0]-C[n+1, i, j-1])/(2*dx)
                        Cww = (C[n+1, i, 0]-2*C[n+1, i, j]+C[n+1, i, j-1])/dx**2
                        C[n, i, j] = C[n+1, i, j]+dt*Ueq(rho[n+1, i], xi[n, i], um, rhom)*Cx-dt/2*(Cx-lamda*Cw)**2-2*dt*lamda*Ueq(rho[n+1, i], xi[n+1, i], um, rhom)*Cw+dt*sigma**2/2*(1-rho[n+1, i])*Cww
                    else:
                        Cx = (C[n+1, i+1, j]-C[n+1, i-1, j])/(2*dx)
                        Cw = (C[n+1, i, j+1]-C[n+1, i, j-1])/(2*dx)
                        Cww = (C[n+1, i, j+1]-2*C[n+1, i, j]+C[n+1, i, j-1])/dx**2
                        C[n, i, j] = C[n+1, i, j]+dt*Ueq(rho[n+1, i], xi[n, i], um, rhom)*Cx-dt/2*(Cx-lamda*Cw)**2-2*dt*lamda*Ueq(rho[n+1, i], xi[n+1, i], um, rhom)*Cw+dt*sigma**2/2*(1-rho[n+1, i])*Cww

    for n in range(Nt):
        for i in range(Nx):
            if i == Nx-1:
                for j in range(Nw):
                    if j == Nw-1:
                        Cx = (C[n, 0, j]-C[n, i-1, j])/(2*dx)
                        Cw = (C[n, i, 0]-C[n, i, j-1])/(2*dw)
                        utilde[n, i, j] = Ueq(rho[n, i], xi[n, i], um, rhom)-(Cx-lamda*Cw)
                    else:
                        Cx = (C[n, 0, j]-C[n, i-1, j])/(2*dx)
                        Cw = (C[n, i, j+1]-C[n, i, j-1])/(2*dw)
                        utilde[n, i, j] = Ueq(rho[n, i], xi[n, i], um, rhom)-(Cx-lamda*Cw)
            else:
                for j in range(Nw):
                    if j == Nw-1:
                        Cx = (C[n, i+1, j]-C[n, i-1, j])/(2*dx)
                        Cw = (C[n, i, 0]-C[n, i, j-1])/(2*dw)
                        utilde[n, i, j] = Ueq(rho[n, i], xi[n, i], um, rhom)-(Cx-lamda*Cw)
                    else:
                        Cx = (C[n, i+1, j]-C[n, i-1, j])/(2*dx)
                        Cw = (C[n, i, j+1]-C[n, i, j-1])/(2*dw)
                        utilde[n, i, j] = Ueq(rho[n, i], xi[n, i], um, rhom)-(Cx-lamda*Cw)
    return C, utilde

def fictitious_play(C, Cold, utilde, utildeold, ii):
    C = 1/(ii+1)*Cold+ii/(ii+1)*C
    utilde = 1/(ii+1)*utildeold+ii/(ii+1)*utildeold
    return C, utilde

Nt = 320
Nx = 80
Nw = 80
t = np.linspace(0, 3, Nt)
x = np.linspace(0, 1, Nx)
w = np.linspace(0, 1, Nw)
dx = x[1]-x[0]
dt = t[1]-t[0]
dw = w[1]-w[0]
um = 1.05
rhom = 1.15
rho0 = 0.9*np.exp(-0.5*(x-0.5)**2/0.05**2)+0.0565
u0 = Ueq(rho0, um, um, rhom)
xi0 = rho0+u0
rho = np.ones((Nt, Nx))
u = np.zeros((Nt, Nx))
xi = u.copy()
C = np.zeros((Nt, Nx, Nw))
utilde = C.copy()
rho[0] = rho0
u[0] = u0
xi[0] = xi0
lamda = 0.001
sigma = 0
max_iter = 100
start = time.time()
for ii in range(max_iter):
    rhonew, xinew = GSOM(rho, xi, u, um, rhom, Nt, Nx, Nw, lamda)
    Cnew, utildenew = HJB(C, u, rhonew, xinew, Nt, Nx, Nw, lamda, sigma)
    Cnew, utildenew = fictitious_play(Cnew, C, utildenew, utilde, ii)
    unew = np.mean(utildenew, axis = 2)
    error = np.linalg.norm(rho-rhonew)+np.linalg.norm(u-unew)+np.linalg.norm(xi-xinew)
    print('At iteration', ii+1, ', the error is:', error)
    clear_output(wait=True)
    rho = rhonew
    xi = xinew
    C = Cnew
    utilde = utildenew
    u = unew
elapsed = time.time()-start
print('Total time elapsed:', elapsed, ' sec.')
print('At iteration', ii+1, ', the error is:', error)