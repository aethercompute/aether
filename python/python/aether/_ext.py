try:
    import _aether_ext
    from _aether_ext import *
except ImportError:
    from ._aether_ext import *
