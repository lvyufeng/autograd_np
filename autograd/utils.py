def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return numpy.asnumpy(x)

def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)