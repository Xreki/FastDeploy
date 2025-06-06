from .xpu import *
from .npu import *

# 合并所有模块的 __all__
__all__ = []
from . import npu
if hasattr(npu, '__all__'):
    __all__.extend(npu.__all__)
    
from . import xpu
if hasattr(xpu, '__all__'):
    __all__.extend(xpu.__all__)