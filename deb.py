from numpy.ctypeslib import ndpointer
import ctypes

sellib = ctypes.CDLL('d:/projects/cpp/indeclib/x64/Release/indeclib.dll')
sellib.get_log_probs.argtypes = [
    ctypes.c_int, ctypes.c_int,
]
sellib.get_log_probs.restype = ctypes.c_float

a, b = 3, 4
c = sellib.cosine_sim_from_vec_list(a, b)
print(c)
