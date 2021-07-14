import matplotlib.pyplot as plt
import numpy as np
import time

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
def mandelbrot0(x, y, nmax=1000):
    n, z = 0, 0 + 0j
    c = complex(x, y)
    while (abs(z) <= 2) and (n < nmax):
        n += 1
        z_nouveau = z**2 + c
        z = z_nouveau
    if n == nmax: 
        return -1
    return n

x1, x2 = 0.02, 0.14
y1, y2 = -0.84, 0.84

x1, x2 = -2, 1
y1, y2 = -1.3, 1.3
Nx, Ny = 300, 300

def go0(x1, x2, y1, y2, Nx, Ny, niter):
    L = np.zeros( (Nx, Ny))
    mi, mj = L.shape
    Lx = np.linspace(x1, x2, Nx)
    Ly = np.linspace(y1, y2, Ny)

    t = time.time()
    for x in range(0, mi):
        for y in range(0, mj):     
            #print(i, Lx[i], j, Ly[i])

            L[y,x ] = mandelbrot0(Lx[x], Ly[y], niter)
            #print(Lx[x], Ly[y], L[x,y])

    print(time.time() - t)
    plt.matshow(L, extent=[x1, x2, y1, y2])
    plt.show()

go0(x1, x2, y1, y2, Nx, Ny, 100)
