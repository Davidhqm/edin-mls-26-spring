"""
Template Baseline - Initial CuPy Configuration
Performance: 3652ms (15.7x slower than PyTorch)

Key Characteristics:
- Pure CuPy tensor operations
- Uses CuPy einsum for attention computation
"""

# Set version-specific configuration
import sys
import os

# Add this directory to path for imports
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

# Import with version-specific settings
from . import layers
layers.Linear.BACKEND = 'cublas'  # Use basic backend for baseline
layers.MLP.FUSED = False
layers.AudioMLP.FUSED = False

# Re-export main modules
from . import model
from . import rope
from . import conv
from . import weight_loader
