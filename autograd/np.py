from autograd.config import GPU

if GPU:
    import cupy as np
else:
    import numpy as np