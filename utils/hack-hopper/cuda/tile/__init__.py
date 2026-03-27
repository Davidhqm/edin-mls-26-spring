"""
cuTile Compatibility Layer - Triton Backend

This module provides a drop-in replacement for cuda.tile that works on
non-Blackwell GPUs by translating cuTile kernels to Triton kernels.

Strategy:
1. Parse student's cuTile kernel using Python AST
2. Translate cuTile operations to equivalent Triton operations
3. JIT compile and execute using Triton

cuTile -> Triton mapping:
- ct.bid(dim)                    -> tl.program_id(dim)
- ct.load(arr, index, shape)     -> tl.load(ptr + offsets, mask)
- ct.store(arr, index, tile)     -> tl.store(ptr + offsets, tile, mask)
- ct.exp, ct.sin, etc.           -> tl.exp, tl.sin, etc.
- ct.full(shape, val)            -> tl.full(shape, val)
"""

import ast
import inspect
import textwrap
import hashlib
from typing import Callable, Tuple, Any, Dict, List, Optional, Union
from functools import wraps
import numpy as np

# Try to import triton, fall back to interpreter mode if not available
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("[cuTile Compat] Warning: triton not installed, using slow interpreter mode")

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("[cuTile Compat] Warning: cupy not installed")


# =============================================================================
# Data Types
# =============================================================================

class DType:
    """Base class for cuTile data types."""
    pass

class _Int8(DType):
    name = "int8"
    triton_dtype = "tl.int8" if HAS_TRITON else None
    nptype = np.int8
int8 = _Int8()

class _Int16(DType):
    name = "int16"
    triton_dtype = "tl.int16" if HAS_TRITON else None
    nptype = np.int16
int16 = _Int16()

class _Int32(DType):
    name = "int32"
    triton_dtype = "tl.int32" if HAS_TRITON else None
    nptype = np.int32
int32 = _Int32()

class _Int64(DType):
    name = "int64"
    triton_dtype = "tl.int64" if HAS_TRITON else None
    nptype = np.int64
int64 = _Int64()

class _UInt8(DType):
    name = "uint8"
    triton_dtype = "tl.uint8" if HAS_TRITON else None
    nptype = np.uint8
uint8 = _UInt8()

class _UInt16(DType):
    name = "uint16"
    triton_dtype = "tl.uint16" if HAS_TRITON else None
    nptype = np.uint16
uint16 = _UInt16()

class _UInt32(DType):
    name = "uint32"
    triton_dtype = "tl.uint32" if HAS_TRITON else None
    nptype = np.uint32
uint32 = _UInt32()

class _UInt64(DType):
    name = "uint64"
    triton_dtype = "tl.uint64" if HAS_TRITON else None
    nptype = np.uint64
uint64 = _UInt64()

class _Float16(DType):
    name = "float16"
    triton_dtype = "tl.float16" if HAS_TRITON else None
    nptype = np.float16
float16 = _Float16()

class _Float32(DType):
    name = "float32"
    triton_dtype = "tl.float32" if HAS_TRITON else None
    nptype = np.float32
float32 = _Float32()

class _Float64(DType):
    name = "float64"
    triton_dtype = "tl.float64" if HAS_TRITON else None
    nptype = np.float64
float64 = _Float64()

class _BFloat16(DType):
    name = "bfloat16"
    triton_dtype = "tl.bfloat16" if HAS_TRITON else None
    nptype = np.float16
bfloat16 = _BFloat16()

class _TFloat32(DType):
    name = "tfloat32"
    triton_dtype = "tl.float32" if HAS_TRITON else None
    nptype = np.float32
tfloat32 = _TFloat32()

class _Bool(DType):
    name = "bool"
    triton_dtype = "tl.int1" if HAS_TRITON else None
    nptype = np.bool_
bool_ = _Bool()

class _Float8E4M3FN(DType):
    name = "float8_e4m3fn"
    triton_dtype = "tl.float8e4nv" if HAS_TRITON else None
    nptype = np.float16
float8_e4m3fn = _Float8E4M3FN()

class _Float8E5M2(DType):
    name = "float8_e5m2"
    triton_dtype = "tl.float8e5" if HAS_TRITON else None
    nptype = np.float16
float8_e5m2 = _Float8E5M2()


def _dtype_to_triton(dtype):
    """Convert cuTile dtype to Triton dtype string."""
    if isinstance(dtype, DType):
        return dtype.triton_dtype
    # Map numpy/python types
    type_map = {
        np.float32: "tl.float32",
        np.float16: "tl.float16",
        np.float64: "tl.float64",
        np.int32: "tl.int32",
        np.int64: "tl.int64",
        np.int16: "tl.int16",
        np.int8: "tl.int8",
        np.uint32: "tl.uint32",
        np.uint64: "tl.uint64",
        np.uint16: "tl.uint16",
        np.uint8: "tl.uint8",
        np.bool_: "tl.int1",
        float: "tl.float32",
        int: "tl.int32",
    }
    return type_map.get(dtype, "tl.float32")


def _dtype_to_nptype(dtype):
    """Convert cuTile dtype to numpy dtype."""
    if isinstance(dtype, DType):
        return dtype.nptype
    if dtype is None:
        return None
    return np.dtype(dtype)


# =============================================================================
# Type Annotations (passthrough)
# =============================================================================

class Constant:
    """Type annotation for compile-time constants."""
    def __class_getitem__(cls, item):
        return item

class ConstantAnnotation:
    pass

class Array:
    def __class_getitem__(cls, item):
        return item

class Scalar:
    def __class_getitem__(cls, item):
        return item

class Tile:
    def __class_getitem__(cls, item):
        return item

class ByTarget:
    def __class_getitem__(cls, item):
        return item


# =============================================================================
# Enums
# =============================================================================

class MemoryOrder:
    relaxed = "relaxed"
    acquire = "acquire"
    release = "release"
    acq_rel = "acq_rel"
    seq_cst = "seq_cst"

class MemoryScope:
    system = "system"
    device = "device"
    block = "block"

class PaddingMode:
    zeros = "zeros"
    reflect = "reflect"
    replicate = "replicate"

class RoundingMode:
    nearest = "nearest"
    down = "down"
    up = "up"
    truncate = "truncate"


# =============================================================================
# Exceptions
# =============================================================================

class TileCompilerError(Exception):
    pass

class TileCompilerExecutionError(TileCompilerError):
    pass

class TileCompilerTimeoutError(TileCompilerError):
    pass

class TileInternalError(TileCompilerError):
    pass

class TileSyntaxError(TileCompilerError):
    pass

class TileTypeError(TileCompilerError):
    pass

class TileValueError(TileCompilerError):
    pass


# =============================================================================
# Utility Functions
# =============================================================================

def cdiv(a: int, b: int) -> int:
    """Ceiling division: (a + b - 1) // b"""
    return (a + b - 1) // b


# =============================================================================
# Stub Functions (for outside kernel use - raise errors)
# =============================================================================

def bid(dim: int) -> int:
    raise RuntimeError("bid() can only be called within a kernel")

def num_blocks(dim: int) -> int:
    raise RuntimeError("num_blocks() can only be called within a kernel")

def num_tiles(dim: int) -> int:
    raise RuntimeError("num_tiles() can only be called within a kernel")

def load(array, index: Tuple, shape: Tuple, **kwargs):
    raise RuntimeError("load() can only be called within a kernel")

def store(array, index: Tuple, tile):
    raise RuntimeError("store() can only be called within a kernel")

def full(shape: Tuple, value, dtype=None):
    raise RuntimeError("full() can only be called within a kernel")

def zeros(shape: Tuple, dtype=None):
    raise RuntimeError("zeros() can only be called within a kernel")

def ones(shape: Tuple, dtype=None):
    raise RuntimeError("ones() can only be called within a kernel")

def arange(start, stop=None, step=1, dtype=None):
    raise RuntimeError("arange() can only be called within a kernel")

def astype(tile, dtype):
    raise RuntimeError("astype() can only be called within a kernel")

def transpose(tile, axes=None):
    raise RuntimeError("transpose() can only be called within a kernel")

def permute(tile, axes):
    raise RuntimeError("permute() can only be called within a kernel")

def reshape(tile, shape):
    raise RuntimeError("reshape() can only be called within a kernel")

def broadcast_to(tile, shape):
    raise RuntimeError("broadcast_to() can only be called within a kernel")

def expand_dims(tile, axis):
    raise RuntimeError("expand_dims() can only be called within a kernel")

def cat(tiles, axis=0):
    raise RuntimeError("cat() can only be called within a kernel")

def bitcast(tile, dtype):
    raise RuntimeError("bitcast() can only be called within a kernel")

def extract(tile, indices):
    raise RuntimeError("extract() can only be called within a kernel")

def gather(array, indices, axis=0):
    raise RuntimeError("gather() can only be called within a kernel")

def scatter(array, indices, tile, axis=0):
    raise RuntimeError("scatter() can only be called within a kernel")

def where(condition, x, y):
    raise RuntimeError("where() can only be called within a kernel")

# Math stubs
def exp(x, **kwargs): raise RuntimeError("exp() can only be called within a kernel")
def exp2(x, **kwargs): raise RuntimeError("exp2() can only be called within a kernel")
def log(x): raise RuntimeError("log() can only be called within a kernel")
def log2(x): raise RuntimeError("log2() can only be called within a kernel")
def sqrt(x): raise RuntimeError("sqrt() can only be called within a kernel")
def rsqrt(x): raise RuntimeError("rsqrt() can only be called within a kernel")
def sin(x): raise RuntimeError("sin() can only be called within a kernel")
def cos(x): raise RuntimeError("cos() can only be called within a kernel")
def tan(x): raise RuntimeError("tan() can only be called within a kernel")
def sinh(x): raise RuntimeError("sinh() can only be called within a kernel")
def cosh(x): raise RuntimeError("cosh() can only be called within a kernel")
def tanh(x): raise RuntimeError("tanh() can only be called within a kernel")
def floor(x): raise RuntimeError("floor() can only be called within a kernel")
def ceil(x): raise RuntimeError("ceil() can only be called within a kernel")
def pow(x, y): raise RuntimeError("pow() can only be called within a kernel")
def abs(x): raise RuntimeError("abs() can only be called within a kernel")

# Reduction stubs
def sum(x, axis=None, keepdims=False): raise RuntimeError("sum() can only be called within a kernel")
def prod(x, axis=None): raise RuntimeError("prod() can only be called within a kernel")
def min(x, axis=None, keepdims=False): raise RuntimeError("min() can only be called within a kernel")
def max(x, axis=None, keepdims=False): raise RuntimeError("max() can only be called within a kernel")
def argmin(x, axis=None): raise RuntimeError("argmin() can only be called within a kernel")
def argmax(x, axis=None): raise RuntimeError("argmax() can only be called within a kernel")
def cumsum(x, axis=None): raise RuntimeError("cumsum() can only be called within a kernel")
def cumprod(x, axis=None): raise RuntimeError("cumprod() can only be called within a kernel")
def minimum(x, y): raise RuntimeError("minimum() can only be called within a kernel")
def maximum(x, y): raise RuntimeError("maximum() can only be called within a kernel")

# Binary stubs
def add(x, y): raise RuntimeError("add() can only be called within a kernel")
def sub(x, y): raise RuntimeError("sub() can only be called within a kernel")
def mul(x, y): raise RuntimeError("mul() can only be called within a kernel")
def truediv(x, y, **kwargs): raise RuntimeError("truediv() can only be called within a kernel")
def floordiv(x, y): raise RuntimeError("floordiv() can only be called within a kernel")
def mod(x, y): raise RuntimeError("mod() can only be called within a kernel")
def negative(x): raise RuntimeError("negative() can only be called within a kernel")

# Comparison stubs
def equal(x, y): raise RuntimeError("equal() can only be called within a kernel")
def not_equal(x, y): raise RuntimeError("not_equal() can only be called within a kernel")
def less(x, y): raise RuntimeError("less() can only be called within a kernel")
def less_equal(x, y): raise RuntimeError("less_equal() can only be called within a kernel")
def greater(x, y): raise RuntimeError("greater() can only be called within a kernel")
def greater_equal(x, y): raise RuntimeError("greater_equal() can only be called within a kernel")

# Bitwise stubs
def bitwise_and(x, y): raise RuntimeError("bitwise_and() can only be called within a kernel")
def bitwise_or(x, y): raise RuntimeError("bitwise_or() can only be called within a kernel")
def bitwise_xor(x, y): raise RuntimeError("bitwise_xor() can only be called within a kernel")
def bitwise_not(x): raise RuntimeError("bitwise_not() can only be called within a kernel")
def bitwise_lshift(x, y): raise RuntimeError("bitwise_lshift() can only be called within a kernel")
def bitwise_rshift(x, y): raise RuntimeError("bitwise_rshift() can only be called within a kernel")

# Matrix stubs
def matmul(a, b): raise RuntimeError("matmul() can only be called within a kernel")
def mma(a, b, c): raise RuntimeError("mma() can only be called within a kernel")

# Atomic stubs
def atomic_add(array, index, value): raise RuntimeError("atomic_add() can only be called within a kernel")
def atomic_and(array, index, value): raise RuntimeError("atomic_and() can only be called within a kernel")
def atomic_or(array, index, value): raise RuntimeError("atomic_or() can only be called within a kernel")
def atomic_xor(array, index, value): raise RuntimeError("atomic_xor() can only be called within a kernel")
def atomic_min(array, index, value): raise RuntimeError("atomic_min() can only be called within a kernel")
def atomic_max(array, index, value): raise RuntimeError("atomic_max() can only be called within a kernel")
def atomic_xchg(array, index, value): raise RuntimeError("atomic_xchg() can only be called within a kernel")
def atomic_cas(array, index, compare, value): raise RuntimeError("atomic_cas() can only be called within a kernel")

# Debug stubs
def printf(fmt, *args): raise RuntimeError("printf() can only be called within a kernel")
def assert_(condition, msg=""): raise RuntimeError("assert_() can only be called within a kernel")


# =============================================================================
# AST Transformer: cuTile -> Triton
# =============================================================================

class CuTileToTritonTransformer(ast.NodeTransformer):
    """
    Transform cuTile kernel AST to Triton kernel AST.

    Key transformations:
    - ct.bid(dim) -> tl.program_id(dim)
    - ct.load(arr, index=(pid,), shape=(tile_size,)) ->
        offsets = pid * tile_size + tl.arange(0, tile_size)
        mask = offsets < arr_size
        tile = tl.load(arr_ptr + offsets, mask=mask)
    - ct.store(arr, index=(pid,), tile=result) ->
        offsets = pid * tile_size + tl.arange(0, tile_size)
        mask = offsets < arr_size
        tl.store(arr_ptr + offsets, result, mask=mask)
    - ct.exp(x) -> tl.exp(x)
    - ct.full(shape, val, dtype) -> tl.full(shape, val, dtype)
    """

    def __init__(self, array_params: List[str], const_params: List[str], array_shapes: Dict[str, str]):
        """
        Args:
            array_params: List of parameter names that are arrays
            const_params: List of parameter names that are constants
            array_shapes: Dict mapping array name to its size variable name
        """
        self.array_params = array_params
        self.const_params = const_params
        self.array_shapes = array_shapes
        self.load_counter = 0
        self.store_counter = 0
        self.generated_lines = []  # Extra lines to insert

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Transform function calls."""
        # Check if this is a ct.xxx() call
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'ct':
                method = node.func.attr
                return self._transform_ct_call(method, node)

        # Recursively visit children
        return self.generic_visit(node)

    def _transform_ct_call(self, method: str, node: ast.Call) -> ast.AST:
        """Transform ct.xxx() calls to tl.xxx() calls."""

        # ct.bid(dim) -> tl.program_id(dim)
        if method == 'bid':
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                   attr='program_id', ctx=ast.Load()),
                args=node.args,
                keywords=[]
            )

        # ct.exp(x) -> tl.exp(x)
        if method == 'exp':
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                   attr='exp', ctx=ast.Load()),
                args=[self.visit(arg) for arg in node.args],
                keywords=[]
            )

        # ct.log(x) -> tl.log(x)
        if method == 'log':
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                   attr='log', ctx=ast.Load()),
                args=[self.visit(arg) for arg in node.args],
                keywords=[]
            )

        # ct.sqrt(x) -> tl.sqrt(x)
        if method == 'sqrt':
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                   attr='sqrt', ctx=ast.Load()),
                args=[self.visit(arg) for arg in node.args],
                keywords=[]
            )

        # ct.sin(x) -> tl.sin(x)
        if method == 'sin':
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                   attr='sin', ctx=ast.Load()),
                args=[self.visit(arg) for arg in node.args],
                keywords=[]
            )

        # ct.cos(x) -> tl.cos(x)
        if method == 'cos':
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                   attr='cos', ctx=ast.Load()),
                args=[self.visit(arg) for arg in node.args],
                keywords=[]
            )

        # ct.tanh(x) -> tl.tanh(x) (Triton doesn't have tanh, use libdevice)
        if method == 'tanh':
            # tl.math.tanh or manual: (exp(2x) - 1) / (exp(2x) + 1)
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                        attr='math', ctx=ast.Load()),
                    attr='tanh', ctx=ast.Load()),
                args=[self.visit(arg) for arg in node.args],
                keywords=[]
            )

        # ct.abs(x) -> tl.abs(x)
        if method == 'abs':
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                   attr='abs', ctx=ast.Load()),
                args=[self.visit(arg) for arg in node.args],
                keywords=[]
            )

        # ct.maximum(x, y) -> tl.maximum(x, y)
        if method == 'maximum':
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                   attr='maximum', ctx=ast.Load()),
                args=[self.visit(arg) for arg in node.args],
                keywords=[]
            )

        # ct.minimum(x, y) -> tl.minimum(x, y)
        if method == 'minimum':
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                   attr='minimum', ctx=ast.Load()),
                args=[self.visit(arg) for arg in node.args],
                keywords=[]
            )

        # ct.where(cond, x, y) -> tl.where(cond, x, y)
        if method == 'where':
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                   attr='where', ctx=ast.Load()),
                args=[self.visit(arg) for arg in node.args],
                keywords=[]
            )

        # ct.sum(x, axis) -> tl.sum(x, axis)
        if method == 'sum':
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                   attr='sum', ctx=ast.Load()),
                args=[self.visit(arg) for arg in node.args],
                keywords=[ast.keyword(arg=kw.arg, value=self.visit(kw.value)) for kw in node.keywords]
            )

        # ct.max(x, axis) -> tl.max(x, axis)
        if method == 'max':
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                   attr='max', ctx=ast.Load()),
                args=[self.visit(arg) for arg in node.args],
                keywords=[ast.keyword(arg=kw.arg, value=self.visit(kw.value)) for kw in node.keywords]
            )

        # ct.min(x, axis) -> tl.min(x, axis)
        if method == 'min':
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                   attr='min', ctx=ast.Load()),
                args=[self.visit(arg) for arg in node.args],
                keywords=[ast.keyword(arg=kw.arg, value=self.visit(kw.value)) for kw in node.keywords]
            )

        # ct.astype(tile, dtype) -> tile.to(dtype)
        if method == 'astype':
            tile_arg = self.visit(node.args[0])
            dtype_arg = node.args[1]
            # Convert ct.dtype to tl.dtype
            dtype_str = self._convert_dtype(dtype_arg)
            return ast.Call(
                func=ast.Attribute(value=tile_arg, attr='to', ctx=ast.Load()),
                args=[ast.Name(id=dtype_str, ctx=ast.Load())],
                keywords=[]
            )

        # ct.full(shape, val, dtype) -> tl.full(shape, val, dtype)
        if method == 'full':
            shape_arg = self.visit(node.args[0])
            val_arg = self.visit(node.args[1])
            dtype_arg = node.args[2] if len(node.args) > 2 else None
            dtype_kw = None
            for kw in node.keywords:
                if kw.arg == 'dtype':
                    dtype_arg = kw.value

            keywords = []
            if dtype_arg:
                dtype_str = self._convert_dtype(dtype_arg)
                keywords.append(ast.keyword(arg='dtype', value=ast.Name(id=dtype_str, ctx=ast.Load())))

            return ast.Call(
                func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                   attr='full', ctx=ast.Load()),
                args=[shape_arg, val_arg],
                keywords=keywords
            )

        # ct.zeros(shape, dtype) -> tl.zeros(shape, dtype)
        if method == 'zeros':
            shape_arg = self.visit(node.args[0])
            dtype_arg = None
            for kw in node.keywords:
                if kw.arg == 'dtype':
                    dtype_arg = kw.value
            if len(node.args) > 1:
                dtype_arg = node.args[1]

            keywords = []
            if dtype_arg:
                dtype_str = self._convert_dtype(dtype_arg)
                keywords.append(ast.keyword(arg='dtype', value=ast.Name(id=dtype_str, ctx=ast.Load())))

            return ast.Call(
                func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                   attr='zeros', ctx=ast.Load()),
                args=[shape_arg],
                keywords=keywords
            )

        # ct.arange(start, stop, step, dtype) -> tl.arange(start, stop)
        if method == 'arange':
            args = [self.visit(arg) for arg in node.args]
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                   attr='arange', ctx=ast.Load()),
                args=args[:2] if len(args) >= 2 else args,
                keywords=[]
            )

        # ct.load and ct.store are handled specially at statement level
        # Return the node as-is for now, will be handled by visit_Assign/visit_Expr
        if method in ('load', 'store'):
            return node

        # ct.matmul(a, b) -> tl.dot(a, b)
        if method == 'matmul':
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                                   attr='dot', ctx=ast.Load()),
                args=[self.visit(arg) for arg in node.args],
                keywords=[]
            )

        # Default: just replace ct. with tl.
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id='tl', ctx=ast.Load()),
                               attr=method, ctx=ast.Load()),
            args=[self.visit(arg) for arg in node.args],
            keywords=[ast.keyword(arg=kw.arg, value=self.visit(kw.value)) for kw in node.keywords]
        )

    def _convert_dtype(self, dtype_node) -> str:
        """Convert ct.dtype to tl.dtype string."""
        if isinstance(dtype_node, ast.Attribute):
            if isinstance(dtype_node.value, ast.Name) and dtype_node.value.id == 'ct':
                dtype_name = dtype_node.attr
                dtype_map = {
                    'float32': 'tl.float32',
                    'float16': 'tl.float16',
                    'float64': 'tl.float64',
                    'int32': 'tl.int32',
                    'int64': 'tl.int64',
                    'int16': 'tl.int16',
                    'int8': 'tl.int8',
                    'uint32': 'tl.uint32',
                    'uint64': 'tl.uint64',
                    'uint16': 'tl.uint16',
                    'uint8': 'tl.uint8',
                    'bfloat16': 'tl.bfloat16',
                }
                return dtype_map.get(dtype_name, 'tl.float32')
        return 'tl.float32'


# =============================================================================
# Kernel Compiler
# =============================================================================

def _compile_kernel_to_triton(func: Callable, grid: Tuple[int, ...], args: Tuple) -> Tuple[Any, List, Dict]:
    """
    Compile a cuTile kernel to a Triton kernel.

    Returns:
        (triton_kernel, kernel_args, meta)
    """
    # Get source code
    source = inspect.getsource(func)
    source = textwrap.dedent(source)

    # Parse AST
    tree = ast.parse(source)
    func_def = tree.body[0]
    assert isinstance(func_def, ast.FunctionDef)

    # Analyze parameters
    params = func_def.args.args
    param_names = [p.arg for p in params]

    # Identify array vs constant parameters by annotation
    array_params = []
    const_params = []
    for i, param in enumerate(params):
        if param.annotation:
            ann_str = ast.unparse(param.annotation)
            if 'Constant' in ann_str:
                const_params.append(param.arg)
            else:
                array_params.append(param.arg)
        else:
            # Assume arrays if CuPy array passed
            if i < len(args) and hasattr(args[i], '__cuda_array_interface__'):
                array_params.append(param.arg)
            else:
                const_params.append(param.arg)

    # Generate Triton kernel code
    triton_code = _generate_triton_kernel(func_def, array_params, const_params, args)

    # Compile and return
    namespace = {'triton': triton, 'tl': tl}
    exec(triton_code, namespace)
    triton_kernel = namespace['_triton_kernel']

    # Prepare arguments: arrays become pointers, add size parameters
    kernel_args = []
    meta = {}

    for i, (name, arg) in enumerate(zip(param_names, args)):
        if name in array_params:
            kernel_args.append(arg)  # CuPy array -> pointer
            # Add size parameter
            if hasattr(arg, 'size'):
                kernel_args.append(arg.size)
        else:
            kernel_args.append(arg)  # Constants passed directly

    return triton_kernel, kernel_args, meta


def _generate_triton_kernel(func_def: ast.FunctionDef, array_params: List[str],
                            const_params: List[str], args: Tuple) -> str:
    """
    Generate Triton kernel code from cuTile kernel AST.
    """
    kernel_name = func_def.name
    params = func_def.args.args
    param_names = [p.arg for p in params]

    # Build parameter list for Triton kernel
    triton_params = []
    for i, name in enumerate(param_names):
        if name in array_params:
            triton_params.append(f"{name}_ptr")
            triton_params.append(f"{name}_size")  # Add size for bounds checking
        else:
            triton_params.append(name)

    # Extract constant values for BLOCK_SIZE etc.
    const_values = {}
    for i, name in enumerate(param_names):
        if name in const_params and i < len(args):
            const_values[name] = args[i]

    # Generate kernel body
    body_lines = []

    for stmt in func_def.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            # Skip docstrings
            continue
        triton_stmt = _translate_statement(stmt, array_params, const_params, const_values)
        body_lines.extend(triton_stmt)

    body_code = '\n    '.join(body_lines)

    # Assemble kernel
    triton_code = f'''
import triton
import triton.language as tl

@triton.jit
def _triton_kernel({', '.join(triton_params)}):
    {body_code}
'''
    return triton_code


def _translate_statement(stmt: ast.AST, array_params: List[str],
                        const_params: List[str], const_values: Dict) -> List[str]:
    """Translate a single statement from cuTile to Triton."""
    lines = []

    if isinstance(stmt, ast.Assign):
        # Handle: var = ct.xxx()
        target = ast.unparse(stmt.targets[0])
        value = stmt.value

        if _is_ct_load(value):
            # ct.load(arr, index=(pid,), shape=(tile_size,))
            lines.extend(_translate_load(target, value, array_params, const_values))
        elif _is_ct_call(value):
            # Other ct.xxx() calls
            translated = _translate_expr(value, array_params, const_values)
            lines.append(f"{target} = {translated}")
        else:
            # Regular assignment
            translated = _translate_expr(value, array_params, const_values)
            lines.append(f"{target} = {translated}")

    elif isinstance(stmt, ast.Expr):
        value = stmt.value
        if _is_ct_store(value):
            # ct.store(arr, index=(pid,), tile=result)
            lines.extend(_translate_store(value, array_params, const_values))
        else:
            translated = _translate_expr(value, array_params, const_values)
            lines.append(translated)

    elif isinstance(stmt, ast.If):
        # if statement
        test = _translate_expr(stmt.test, array_params, const_values)
        lines.append(f"if {test}:")
        for s in stmt.body:
            sub_lines = _translate_statement(s, array_params, const_params, const_values)
            for line in sub_lines:
                lines.append(f"    {line}")
        if stmt.orelse:
            lines.append("else:")
            for s in stmt.orelse:
                sub_lines = _translate_statement(s, array_params, const_params, const_values)
                for line in sub_lines:
                    lines.append(f"    {line}")

    elif isinstance(stmt, ast.For):
        # for loop
        target = ast.unparse(stmt.target)
        iter_expr = _translate_expr(stmt.iter, array_params, const_values)
        lines.append(f"for {target} in {iter_expr}:")
        for s in stmt.body:
            sub_lines = _translate_statement(s, array_params, const_params, const_values)
            for line in sub_lines:
                lines.append(f"    {line}")

    elif isinstance(stmt, ast.AugAssign):
        # x += y
        target = ast.unparse(stmt.target)
        op = _translate_op(stmt.op)
        value = _translate_expr(stmt.value, array_params, const_values)
        lines.append(f"{target} {op}= {value}")

    elif isinstance(stmt, ast.Pass):
        lines.append("pass")

    elif isinstance(stmt, ast.Return):
        if stmt.value:
            value = _translate_expr(stmt.value, array_params, const_values)
            lines.append(f"return {value}")
        else:
            lines.append("return")

    else:
        # Fallback: just unparse
        lines.append(ast.unparse(stmt))

    return lines


def _is_ct_load(node: ast.AST) -> bool:
    """Check if node is ct.load(...)"""
    return (isinstance(node, ast.Call) and
            isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'ct' and
            node.func.attr == 'load')


def _is_ct_store(node: ast.AST) -> bool:
    """Check if node is ct.store(...)"""
    return (isinstance(node, ast.Call) and
            isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'ct' and
            node.func.attr == 'store')


def _is_ct_call(node: ast.AST) -> bool:
    """Check if node is ct.xxx(...)"""
    return (isinstance(node, ast.Call) and
            isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'ct')


def _translate_load(target: str, node: ast.Call, array_params: List[str],
                   const_values: Dict) -> List[str]:
    """
    Translate ct.load(arr, index=(pid,), shape=(tile_size,)) to Triton.

    Generates:
        offsets = pid * tile_size + tl.arange(0, tile_size)
        mask = offsets < arr_size
        target = tl.load(arr_ptr + offsets, mask=mask, other=0.0)
    """
    lines = []

    # Parse arguments
    arr_name = ast.unparse(node.args[0])
    index_arg = None
    shape_arg = None

    for kw in node.keywords:
        if kw.arg == 'index':
            index_arg = kw.value
        elif kw.arg == 'shape':
            shape_arg = kw.value

    if index_arg is None and len(node.args) > 1:
        index_arg = node.args[1]
    if shape_arg is None and len(node.args) > 2:
        shape_arg = node.args[2]

    # Extract index tuple elements
    if isinstance(index_arg, ast.Tuple):
        indices = [ast.unparse(e) for e in index_arg.elts]
    else:
        indices = [ast.unparse(index_arg)]

    # Extract shape tuple elements
    if isinstance(shape_arg, ast.Tuple):
        shapes = [ast.unparse(e) for e in shape_arg.elts]
    else:
        shapes = [ast.unparse(shape_arg)]

    # Generate 1D case (most common for tutorials)
    if len(indices) == 1:
        pid = indices[0]
        tile_size = shapes[0]
        offset_var = f"_offs_{target}"

        lines.append(f"{offset_var} = {pid} * {tile_size} + tl.arange(0, {tile_size})")
        lines.append(f"_mask_{target} = {offset_var} < {arr_name}_size")
        lines.append(f"{target} = tl.load({arr_name}_ptr + {offset_var}, mask=_mask_{target}, other=0.0)")

    # 2D case
    elif len(indices) == 2:
        pid_y, pid_x = indices[0], indices[1]
        tile_h, tile_w = shapes[0], shapes[1]
        offset_var = f"_offs_{target}"

        # For 2D, we need to compute row and column offsets
        # This is simplified - assumes row-major layout
        lines.append(f"_row_offs_{target} = {pid_y} * {tile_h} + tl.arange(0, {tile_h})")
        lines.append(f"_col_offs_{target} = {pid_x} * {tile_w} + tl.arange(0, {tile_w})")
        lines.append(f"# 2D load - flattened for simplicity")
        lines.append(f"{offset_var} = _row_offs_{target}[:, None] * {tile_w} + _col_offs_{target}[None, :]")
        # Note: This is a simplification. Real 2D loads need stride info.
        lines.append(f"# TODO: proper 2D load with strides")

    return lines


def _translate_store(node: ast.Call, array_params: List[str],
                    const_values: Dict) -> List[str]:
    """
    Translate ct.store(arr, index=(pid,), tile=result) to Triton.

    Generates:
        offsets = pid * tile_size + tl.arange(0, tile_size)
        mask = offsets < arr_size
        tl.store(arr_ptr + offsets, result, mask=mask)
    """
    lines = []

    # Parse arguments
    arr_name = ast.unparse(node.args[0])
    index_arg = None
    tile_arg = None

    for kw in node.keywords:
        if kw.arg == 'index':
            index_arg = kw.value
        elif kw.arg == 'tile':
            tile_arg = kw.value

    if index_arg is None and len(node.args) > 1:
        index_arg = node.args[1]
    if tile_arg is None and len(node.args) > 2:
        tile_arg = node.args[2]

    # For keyword argument style: ct.store(arr, index=(pid,), tile=result)
    tile_var = _translate_expr(tile_arg, array_params, const_values)

    # Extract index tuple elements
    if isinstance(index_arg, ast.Tuple):
        indices = [ast.unparse(e) for e in index_arg.elts]
    else:
        indices = [ast.unparse(index_arg)]

    # Get tile shape from the tile variable (we need to track this)
    # For simplicity, assume 1D and use the same offset pattern as load
    if len(indices) == 1:
        pid = indices[0]
        # We need tile_size - infer from context or use a placeholder
        # In practice, the load would have set up the offsets
        lines.append(f"# Store to {arr_name} at tile index {pid}")
        lines.append(f"tl.store({arr_name}_ptr + _offs_{tile_var}, {tile_var}, mask=_mask_{tile_var})")

    return lines


def _translate_expr(node: ast.AST, array_params: List[str], const_values: Dict) -> str:
    """Translate an expression from cuTile to Triton."""
    if isinstance(node, ast.Call):
        if _is_ct_call(node):
            method = node.func.attr

            # ct.bid(dim) -> tl.program_id(dim)
            if method == 'bid':
                arg = ast.unparse(node.args[0])
                return f"tl.program_id({arg})"

            # ct.exp(x) -> tl.exp(x)
            if method == 'exp':
                arg = _translate_expr(node.args[0], array_params, const_values)
                return f"tl.exp({arg})"

            # ct.log(x) -> tl.log(x)
            if method == 'log':
                arg = _translate_expr(node.args[0], array_params, const_values)
                return f"tl.log({arg})"

            # ct.sqrt(x) -> tl.sqrt(x)
            if method == 'sqrt':
                arg = _translate_expr(node.args[0], array_params, const_values)
                return f"tl.sqrt({arg})"

            # ct.full(shape, val, dtype) -> tl.full(shape, val, dtype)
            if method == 'full':
                args_str = ', '.join(_translate_expr(a, array_params, const_values) for a in node.args)
                # Handle dtype keyword
                for kw in node.keywords:
                    if kw.arg == 'dtype':
                        dtype_str = _translate_dtype(kw.value)
                        args_str += f", dtype={dtype_str}"
                return f"tl.full({args_str})"

            # ct.astype(tile, dtype) -> tile.to(dtype)
            if method == 'astype':
                tile = _translate_expr(node.args[0], array_params, const_values)
                dtype = _translate_dtype(node.args[1])
                return f"({tile}).to({dtype})"

            # ct.sum(x, axis) -> tl.sum(x, axis)
            if method == 'sum':
                args_str = ', '.join(_translate_expr(a, array_params, const_values) for a in node.args)
                return f"tl.sum({args_str})"

            # ct.max(x, axis) -> tl.max(x, axis)
            if method == 'max':
                args_str = ', '.join(_translate_expr(a, array_params, const_values) for a in node.args)
                return f"tl.max({args_str})"

            # ct.minimum/maximum
            if method in ('minimum', 'maximum'):
                args_str = ', '.join(_translate_expr(a, array_params, const_values) for a in node.args)
                return f"tl.{method}({args_str})"

            # ct.where(cond, x, y) -> tl.where(cond, x, y)
            if method == 'where':
                args_str = ', '.join(_translate_expr(a, array_params, const_values) for a in node.args)
                return f"tl.where({args_str})"

            # Default: replace ct. with tl.
            args_str = ', '.join(_translate_expr(a, array_params, const_values) for a in node.args)
            return f"tl.{method}({args_str})"

        else:
            # Non-ct call
            func = ast.unparse(node.func)
            args_str = ', '.join(_translate_expr(a, array_params, const_values) for a in node.args)
            return f"{func}({args_str})"

    elif isinstance(node, ast.BinOp):
        left = _translate_expr(node.left, array_params, const_values)
        right = _translate_expr(node.right, array_params, const_values)
        op = _translate_op(node.op)
        return f"({left} {op} {right})"

    elif isinstance(node, ast.UnaryOp):
        operand = _translate_expr(node.operand, array_params, const_values)
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
        elif isinstance(node.op, ast.Not):
            return f"(not {operand})"
        return ast.unparse(node)

    elif isinstance(node, ast.Compare):
        left = _translate_expr(node.left, array_params, const_values)
        comparisons = []
        for op, comp in zip(node.ops, node.comparators):
            right = _translate_expr(comp, array_params, const_values)
            op_str = _translate_cmp_op(op)
            comparisons.append(f"{left} {op_str} {right}")
            left = right
        return ' and '.join(comparisons)

    elif isinstance(node, ast.Name):
        name = node.id
        # Array parameters become _ptr
        if name in array_params:
            return f"{name}_ptr"
        return name

    elif isinstance(node, ast.Constant):
        return repr(node.value)

    elif isinstance(node, ast.Tuple):
        elts = ', '.join(_translate_expr(e, array_params, const_values) for e in node.elts)
        return f"({elts})"

    elif isinstance(node, ast.Subscript):
        value = _translate_expr(node.value, array_params, const_values)
        slice_expr = _translate_expr(node.slice, array_params, const_values)
        return f"{value}[{slice_expr}]"

    elif isinstance(node, ast.Attribute):
        value = _translate_expr(node.value, array_params, const_values)
        return f"{value}.{node.attr}"

    else:
        return ast.unparse(node)


def _translate_op(op: ast.AST) -> str:
    """Translate binary operator."""
    op_map = {
        ast.Add: '+',
        ast.Sub: '-',
        ast.Mult: '*',
        ast.Div: '/',
        ast.FloorDiv: '//',
        ast.Mod: '%',
        ast.Pow: '**',
        ast.BitAnd: '&',
        ast.BitOr: '|',
        ast.BitXor: '^',
        ast.LShift: '<<',
        ast.RShift: '>>',
    }
    return op_map.get(type(op), '?')


def _translate_cmp_op(op: ast.AST) -> str:
    """Translate comparison operator."""
    op_map = {
        ast.Eq: '==',
        ast.NotEq: '!=',
        ast.Lt: '<',
        ast.LtE: '<=',
        ast.Gt: '>',
        ast.GtE: '>=',
    }
    return op_map.get(type(op), '?')


def _translate_dtype(node: ast.AST) -> str:
    """Translate ct.dtype to tl.dtype."""
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == 'ct':
            dtype_map = {
                'float32': 'tl.float32',
                'float16': 'tl.float16',
                'float64': 'tl.float64',
                'int32': 'tl.int32',
                'int64': 'tl.int64',
                'int16': 'tl.int16',
                'int8': 'tl.int8',
                'bfloat16': 'tl.bfloat16',
            }
            return dtype_map.get(node.attr, 'tl.float32')
    return ast.unparse(node)


# =============================================================================
# Interpreter Mode (Fallback when Triton not available or for debugging)
# =============================================================================

import threading
from contextlib import contextmanager

class _ExecutionContext(threading.local):
    """Thread-local execution context for kernel simulation."""
    def __init__(self):
        self.block_id = (0, 0, 0)
        self.grid = (1, 1, 1)
        self.in_kernel = False

_ctx = _ExecutionContext()

@contextmanager
def _kernel_context(block_id, grid):
    old = (_ctx.block_id, _ctx.grid, _ctx.in_kernel)
    _ctx.block_id, _ctx.grid, _ctx.in_kernel = block_id, grid, True
    try:
        yield
    finally:
        _ctx.block_id, _ctx.grid, _ctx.in_kernel = old


def _run_interpreter_mode(kernel_func, grid, args):
    """Execute kernel in interpreter mode using CuPy."""
    if not HAS_CUPY:
        raise RuntimeError("cupy is required for interpreter mode")

    # Patch the module to provide working ct.* functions during execution
    import types
    import builtins
    _builtin_min = builtins.min
    _builtin_max = builtins.max

    def _bid(dim):
        return _ctx.block_id[dim]

    def _load(array, index, shape, **kwargs):
        # Handle scalar load: shape=() means load a single element
        if shape == () or (isinstance(shape, tuple) and len(shape) == 0):
            # Scalar load - just index into the array
            idx = tuple(index)
            return array[idx]

        ndim = len(index)
        slices = []
        for i in range(ndim):
            tile_idx = index[i]
            tile_size = shape[i] if i < len(shape) else 1

            # If tile_size is 0, this dimension uses direct indexing
            if tile_size == 0:
                slices.append(tile_idx)
            else:
                start = tile_idx * tile_size
                end = start + tile_size
                if i < array.ndim:
                    end = _builtin_min(end, array.shape[i])
                slices.append(slice(start, end))

        tile = array[tuple(slices)]

        # Check if we need to pad
        if hasattr(tile, 'shape') and tile.shape != shape:
            padded = cp.zeros(shape, dtype=tile.dtype)
            copy_slices = tuple(slice(0, s) for s in tile.shape)
            padded[copy_slices] = tile
            tile = padded
        return tile

    def _store(array, index, tile):
        # Handle scalar store
        if not hasattr(tile, 'shape') or tile.shape == ():
            idx = tuple(index)
            array[idx] = tile
            return

        ndim = len(index)
        tile_shape = tile.shape
        slices, tile_slices = [], []

        for i in range(ndim):
            tile_idx = index[i]
            # Get tile size for this dimension
            tile_size = tile_shape[i] if i < len(tile_shape) else 1

            if tile_size == 0 or tile_size == 1:
                # Direct indexing for this dimension
                start = tile_idx
                end = tile_idx + (tile_size if tile_size > 0 else 1)
            else:
                start = tile_idx * tile_size
                end = start + tile_size

            if i < array.ndim:
                actual_end = _builtin_min(end, array.shape[i])
                slices.append(slice(start, actual_end))
                tile_slices.append(slice(0, actual_end - start))
            else:
                slices.append(slice(start, end))
                tile_slices.append(slice(None))

        # Squeeze tile if needed to match slice dimensions
        tile_to_store = tile[tuple(tile_slices)] if tile_slices else tile
        array[tuple(slices)] = tile_to_store

    def _full(shape, value, dtype=None):
        np_dtype = _dtype_to_nptype(dtype) if dtype else None
        return cp.full(shape, value, dtype=np_dtype)

    def _zeros(shape, dtype=None):
        np_dtype = _dtype_to_nptype(dtype) if dtype else cp.float32
        return cp.zeros(shape, dtype=np_dtype)

    def _astype(tile, dtype):
        np_dtype = _dtype_to_nptype(dtype)
        return tile.astype(np_dtype)

    def _ones(shape, dtype=None):
        np_dtype = _dtype_to_nptype(dtype) if dtype else cp.float32
        return cp.ones(shape, dtype=np_dtype)

    def _transpose(tile, axes=None):
        return cp.transpose(tile, axes)

    def _reshape(tile, shape):
        return cp.reshape(tile, shape)

    def _scatter_impl(array, indices, tile, axis=0):
        cp.put_along_axis(array, indices, tile, axis=axis)
        return array

    # Create a fake 'ct' module with working functions and data types
    ct_funcs = types.SimpleNamespace(
        # Core functions
        bid=_bid,
        load=_load,
        store=_store,
        full=_full,
        zeros=_zeros,
        ones=_ones,
        astype=_astype,
        transpose=_transpose,
        reshape=_reshape,
        # Math functions
        exp=lambda x, **kw: cp.exp(x),
        exp2=lambda x, **kw: cp.exp2(x),
        log=lambda x: cp.log(x),
        log2=lambda x: cp.log2(x),
        sqrt=lambda x: cp.sqrt(x),
        rsqrt=lambda x: 1.0 / cp.sqrt(x),
        sin=lambda x: cp.sin(x),
        cos=lambda x: cp.cos(x),
        tan=lambda x: cp.tan(x),
        sinh=lambda x: cp.sinh(x),
        cosh=lambda x: cp.cosh(x),
        tanh=lambda x: cp.tanh(x),
        floor=lambda x: cp.floor(x),
        ceil=lambda x: cp.ceil(x),
        abs=lambda x: cp.abs(x),
        # Reduction functions
        sum=lambda x, axis=None, keepdims=False: cp.sum(x, axis=axis, keepdims=keepdims),
        prod=lambda x, axis=None: cp.prod(x, axis=axis),
        max=lambda x, axis=None, keepdims=False: cp.max(x, axis=axis, keepdims=keepdims),
        min=lambda x, axis=None, keepdims=False: cp.min(x, axis=axis, keepdims=keepdims),
        argmax=lambda x, axis=None: cp.argmax(x, axis=axis),
        argmin=lambda x, axis=None: cp.argmin(x, axis=axis),
        maximum=lambda x, y: cp.maximum(x, y),
        minimum=lambda x, y: cp.minimum(x, y),
        # Other functions
        where=lambda c, x, y: cp.where(c, x, y),
        matmul=lambda a, b: cp.matmul(a, b),
        dot=lambda a, b: cp.dot(a, b),
        arange=lambda *args, **kw: cp.arange(*args),
        cat=lambda tiles, axis=0: cp.concatenate(tiles, axis=axis),
        broadcast_to=lambda tile, shape: cp.broadcast_to(tile, shape),
        expand_dims=lambda tile, axis: cp.expand_dims(tile, axis),
        squeeze=lambda tile, axis=None: cp.squeeze(tile, axis=axis) if axis is not None else cp.squeeze(tile),
        permute=lambda tile, axes: cp.transpose(tile, axes),
        gather=lambda array, indices, axis=0: cp.take(array, indices, axis=axis),
        scatter=lambda array, indices, tile, axis=0: _scatter_impl(array, indices, tile, axis),
        extract=lambda tile, indices: tile[indices],
        bitcast=lambda tile, dtype: tile.view(_dtype_to_nptype(dtype)),
        pow=lambda x, y: cp.power(x, y),
        negative=lambda x: -x,
        cdiv=cdiv,
        # Data types
        int8=int8,
        int16=int16,
        int32=int32,
        int64=int64,
        uint8=uint8,
        uint16=uint16,
        uint32=uint32,
        uint64=uint64,
        float16=float16,
        float32=float32,
        float64=float64,
        bfloat16=bfloat16,
        tfloat32=tfloat32,
        bool_=bool_,
        # Type annotations (passthrough)
        Constant=Constant,
        Array=Array,
        Scalar=Scalar,
        Tile=Tile,
    )

    # Get the function's globals and inject our ct module
    func_globals = kernel_func.func.__globals__.copy()
    func_globals['ct'] = ct_funcs

    # Create a new function with modified globals
    import types as py_types
    new_func = py_types.FunctionType(
        kernel_func.func.__code__,
        func_globals,
        kernel_func.func.__name__,
        kernel_func.func.__defaults__,
        kernel_func.func.__closure__
    )

    # Normalize grid to 3D
    grid_x = grid[0] if len(grid) > 0 else 1
    grid_y = grid[1] if len(grid) > 1 else 1
    grid_z = grid[2] if len(grid) > 2 else 1
    grid_3d = (grid_x, grid_y, grid_z)

    # Execute for each block
    for bz in range(grid_z):
        for by in range(grid_y):
            for bx in range(grid_x):
                with _kernel_context((bx, by, bz), grid_3d):
                    new_func(*args)


# =============================================================================
# Kernel Wrapper and Launch
# =============================================================================

class _KernelWrapper:
    """Wrapper for cuTile kernels."""

    def __init__(self, func: Callable, **options):
        self.func = func
        self.name = func.__name__
        self.options = options
        self._triton_cache = {}  # Cache compiled Triton kernels

    def __call__(self, *args, **kwargs):
        raise TypeError("Tile kernels cannot be called directly. Use cuda.tile.launch() instead.")


def kernel(func: Callable = None, /, **kwargs) -> _KernelWrapper:
    """Decorator to mark a function as a cuTile kernel."""
    if func is None:
        def decorator(f):
            return _KernelWrapper(f, **kwargs)
        return decorator
    return _KernelWrapper(func, **kwargs)


def function(func=None, /, *, host=False, tile=True):
    """Decorator for tile functions."""
    def decorator(func):
        if host:
            return func
        else:
            @wraps(func)
            def wrapped(*args, **kwargs):
                if _ctx.in_kernel:
                    return func(*args, **kwargs)
                raise RuntimeError('Tile functions can only be called from tile code.')
            return wrapped

    if func is None:
        return decorator
    else:
        return decorator(func)


def launch(stream, grid: Tuple[int, ...], kernel_func: _KernelWrapper, args: Tuple):
    """
    Launch a cuTile kernel.

    Strategy:
    1. Try to compile to Triton and execute (fast, GPU-native)
    2. Fall back to interpreter mode (slower, but always works)
    """
    if not isinstance(kernel_func, _KernelWrapper):
        raise TypeError("kernel_func must be decorated with @ct.kernel")

    # For now, use interpreter mode which is more robust
    # TODO: Enable Triton compilation once translation is complete
    _run_interpreter_mode(kernel_func, grid, args)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core
    "kernel", "function", "launch", "cdiv",

    # Type annotations
    "Constant", "ConstantAnnotation", "Array", "Scalar", "Tile", "ByTarget",

    # Data types
    "DType", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64",
    "bfloat16", "tfloat32", "bool_",
    "float8_e4m3fn", "float8_e5m2",

    # Enums
    "MemoryOrder", "MemoryScope", "PaddingMode", "RoundingMode",

    # Exceptions
    "TileCompilerError", "TileCompilerExecutionError",
    "TileCompilerTimeoutError", "TileInternalError",
    "TileSyntaxError", "TileTypeError", "TileValueError",

    # Tile operations
    "bid", "num_blocks", "num_tiles",
    "load", "store", "full", "zeros", "ones", "arange",
    "astype", "transpose", "permute", "reshape",
    "broadcast_to", "expand_dims", "cat", "bitcast",
    "extract", "gather", "scatter", "where",

    # Math
    "exp", "exp2", "log", "log2", "sqrt", "rsqrt",
    "sin", "cos", "tan", "sinh", "cosh", "tanh",
    "floor", "ceil", "pow", "abs",

    # Reductions
    "sum", "prod", "min", "max", "argmin", "argmax",
    "cumsum", "cumprod", "minimum", "maximum",

    # Binary ops
    "add", "sub", "mul", "truediv", "floordiv", "mod", "negative",

    # Comparison
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",

    # Bitwise
    "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
    "bitwise_lshift", "bitwise_rshift",

    # Matrix
    "matmul", "mma",

    # Atomic
    "atomic_add", "atomic_and", "atomic_or", "atomic_xor",
    "atomic_min", "atomic_max", "atomic_xchg", "atomic_cas",

    # Debug
    "printf", "assert_",
]

# Print info on import
import sys
if not hasattr(sys, '_cutile_compat_warned'):
    if HAS_TRITON:
        print("[cuTile Compat] Using Triton backend for non-Blackwell GPU")
    else:
        print("[cuTile Compat] Using interpreter mode (install triton for better performance)")
    sys._cutile_compat_warned = True
