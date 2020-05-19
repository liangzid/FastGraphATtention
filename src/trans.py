import torch
from fastGAT.model import FastGAT


traced_script_module=FastGAT.jit.trace(FastGAT,)





