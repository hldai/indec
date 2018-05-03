from numpy.ctypeslib import ndpointer
import ctypes

t = 0
for i in range(21):
    m = t % 60
    h = t // 60 + 12
    m1 = (t + 10) % 60
    h1 = (t + 10) // 60 + 12
    print('{}, {}:{:02d}-{}:{:02d}'.format(i + 1, h, m, h1, m1))
    t += 10
