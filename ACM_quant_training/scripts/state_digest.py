"""Tensor/optimizer/RNG identity digest retained from the completed method."""
import hashlib
import numpy as np
import torch

def digest(obj):
 h=hashlib.sha256()
 def visit(x):
  if torch.is_tensor(x):h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
  elif isinstance(x,np.ndarray):h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
  elif isinstance(x,dict):
   for k in sorted(x,key=str):h.update(repr(k).encode());visit(x[k])
  elif isinstance(x,(list,tuple)):
   h.update(str(type(x)).encode())
   for v in x:visit(v)
  else:h.update(repr(x).encode())
 visit(obj);return h.hexdigest()
