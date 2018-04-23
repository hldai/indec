from numpy.ctypeslib import ndpointer
import ctypes

sellib = ctypes.CDLL('d:/projects/cpp/indeclib/x64/Release/indeclib.dll')
sellib.cosine_sim_from_vec_list.argtypes = [
    ctypes.c_int, ctypes.c_int,
]
sellib.cosine_sim_from_vec_list.restype = ctypes.c_float

a, b = 3, 4
c = sellib.cosine_sim_from_vec_list(a, b)
print(c)
