import matplotlib.pyplot as plt



def mandelbrot1(x, y, nmax=100):
    n, za, zb = 0, 0, 0
    modulec = 0

    while (modulec < 4.0) and (n < nmax):
        n += 1
        za =+ x
        zb =+ y
        zca = za**2 - zb**2
        zcb = 2*za*zb

        modulec = za**2 + zb**2

    return n
