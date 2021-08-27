#%%
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit, njit, prange, vectorize
import imageio
def mandelbrot1(x, y, nmax=100):
    n, za, zb = 0, 0, 0
    modulec = 0

    while (modulec <= 4.0) and (n < nmax):
        n += 1

        zca = za**2 - zb**2
        zcb = 2*za*zb
        zca =+ x
        zcb =+ y
        modulec = zca**2 + zcb**2

    return n
@njit
def mandelbrot0(x, y, nmax=50):
    n, z = 0, 0 + 0j
    c = complex(x, y)
    while (abs(z) <= 2.) and (n < nmax):
        n += 1
        z_nouveau = z**2 + c
        z = z_nouveau
    if n == nmax: 
        return 0
    return int(n)
#@profile
def mandelbrot2(x, y, nmax=50):
    an, bn = 0., 0.
    modc = 0.
    n = 0
    while (modc <= 4.0) and (n < nmax):
        n += 1
        an, bn = (an*an) - (bn*bn) + x, (2*bn*an) + y

        modc = (an*an) + (bn*bn)

    if n == nmax:
        return -1
    return n

def mandelbrot_matrice(Lx, Ly, niter):
    Nx, Ny = len(Lx), len(Ly)
    A, B = np.meshgrid(Lx, Ly)
    C = A + 1j*B
    Z = np.zeros(C.shape, dtype=complex)#np.copy(C)
    N = np.zeros( Z.shape, dtype=int)
    F = np.full( Z.shape, True, dtype=bool)
    for n in range(0, niter):
        Z[F] = Z[F]*Z[F] + C[F]
        M = np.abs(Z)
        N[M <= 2] +=1
        F[M > 2] = False

    N[ N == niter] = -1
    return N
@njit #"""https://github.com/numba/numba/issues/6416"""
def make_zeros(J, N):
    return np.zeros((J[()], N[()]))
@njit
def make_full(J, N, v):
    return np.full((J[()], N[()]), v)

@jit(nopython=True, parallel=False)
def mandelbrot_matrice_numba(Lx, Ly,C,  niter):
    Z = np.copy(C)
    Nx = np.zeros(shape=(), dtype=np.int64)
    Ny = np.zeros(shape=(), dtype=np.int64)
    Nx[()] = len(Lx)
    Ny[()] = len(Ly)
    N = make_zeros(Nx, Ny)
    F = make_full(Nx, Ny, True)
    for n in range(0, niter):
        Z[F] = Z[F]*Z[F] + C[F]
        M = np.abs(Z)
        N[M <= 2] +=1
        F[M > 2] = False

    N[ N == niter] = -1
    return N

x1, x2 = 0.02, 0.14
y1, y2 = -0.84, 0.84

x1, x2 = -2, 1
y1, y2 = -1.3, 1.3
Nx, Ny = 10000, 10000
@njit(parallel=True)
def go(fonction, x1, x2, y1, y2, Nx, Ny, niter):
    L = np.zeros( (Nx, Ny))
    mi, mj = L.shape
    Lx = np.linspace(x1, x2, Nx)
    Ly = np.linspace(y1, y2, Ny)

    for x in prange(0, mi):
        for y in prange(0, mj):     
            #print(i, Lx[i], j, Ly[i])

            L[y,x ] = fonction(Lx[x], Ly[y], niter)
            #print(Lx[x], Ly[y], L[x,y])

    return L

def t():
    go(mandelbrot0, x1, x2, y1, y2, Nx, Ny, 300)
def t2():
    go(mandelbrot2, x1, x2, y1, y2, Nx, Ny, 100)

#t()

def goM(x1, x2, y1, y2, Nx, Ny, niter):
    L = np.zeros( (Nx, Ny))
    mi, mj = L.shape
    Lx = np.linspace(x1, x2, Nx)
    Ly = np.linspace(y1, y2, Ny)

    t = time.time()
    N = mandelbrot_matrice(Lx, Ly, niter)

    print(time.time() - t)
    plt.matshow(N, extent=[x1, x2, y1, y2])
    plt.show()
    return N

def goMN(x1, x2, y1, y2, Nx, Ny, niter):
    L = np.zeros( (Nx, Ny))
    mi, mj = L.shape
    Lx = np.linspace(x1, x2, Nx)
    Ly = np.linspace(y1, y2, Ny)
    A, B = np.meshgrid(Lx, Ly)
    C = A + 1j*B

    N = mandelbrot_matrice_numba(Lx, Ly,C,  niter)

    return N
#t1 = time.time()
#A1 = go(mandelbrot0, x1, x2, y1, y2, Nx, Ny, 100)
#print( time.time() - t1)
#imageio.imwrite("mdb.png", A1)
#plt.imsave('mdb2.png', A1)
#plt.matshow(A1, extent=[x1, x2, y1, y2])
#plt.show()

#A2 = goM(x1, x2, y1, y2, Nx, Ny, 100)

#print(np.array_equal(A1, A2))

def go2(x1, x2, y1, y2, Nx, Ny, niter):
    L = np.zeros( (Nx, Ny))
    Lx = np.linspace(x1, x2, Nx)
    Ly = np.linspace(y1, y2, Ny)
    A, B = np.meshgrid(Lx, Ly)
    C = A + 1j*B

    N = mb2(C)

    return N

@vectorize(["int32(complex128)"], nopython=True, target="parallel")
def mb2(z):
    c = z
    n, nmax = 1, 250
    while (abs(z) <= 2.) and (n < nmax):
        n += 1
        z = z**2 + c
    if n == nmax: 
        return 0
    return n




#%%
t1 = time.time()
A3 = go2(x1, x2, y1, y2, Nx, Ny, 100)
print( time.time() - t1)
plt.matshow(A3, extent=[x1, x2, y1, y2], cmap="turbo")
plt.show()
#%%
t1 = time.time()
A1 = go(mandelbrot0, x1, x2, y1, y2, Nx, Ny, 250)
print( time.time() - t1)
#imageio.imwrite("mdb.png", A1)
#plt.imsave('mdb2.png', A1)
plt.matshow(A1, extent=[x1, x2, y1, y2], cmap="turbo")
plt.show()

#A2 = goM(x1, x2, y1, y2, Nx, Ny, 100)

# %%
