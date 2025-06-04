import numpy as np
import torch

def aligned_tensor(size, dtype=np.float16, alignment=4096):
    itemsize = np.dtype(dtype).itemsize
    extra = alignment // itemsize

    buf = np.empty(size + extra, dtype=dtype)

    address = buf.ctypes.data
    offset = (-address % alignment) // itemsize
    aligned_buf = buf[offset:offset + size]

    assert aligned_buf.ctypes.data % alignment == 0, "Alignment failed"
    return torch.from_numpy(aligned_buf).contiguous()
