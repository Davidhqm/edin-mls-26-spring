#!/usr/bin/env bash
set -eo pipefail

# =========================
# Config
# =========================
ENV_NAME="mls"  # Machine Learning Systems - shared by cutile-tutorial and hw1
PYTHON_VERSION="3.11"
CUDA_TAG="cuda13x"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_INSTALL_DIR="${HOME}/miniconda3"
CONDA_ACTIVATE=""

# Parse command line arguments
AUTO_YES=false
while [[ $# -gt 0 ]]; do
	case $1 in
		-y|--yes)
			AUTO_YES=true
			shift
			;;
		-h|--help)
			echo "Usage: $0 [OPTIONS]"
			echo ""
			echo "Options:"
			echo "  -y, --yes    Non-interactive mode, answer yes to all prompts"
			echo "  -h, --help   Show this help message"
			exit 0
			;;
		*)
			echo "Unknown option: $1"
			exit 1
			;;
	esac
done

# =========================
# Helper functions
# =========================
ask_continue() {
	local prompt="${1:-Continue?}"
	if [ "${AUTO_YES}" = true ]; then
		echo ">>> ${prompt} [Y/n] y (auto)"
		return 0
	fi
	read -rp ">>> ${prompt} [Y/n] " answer
	case "${answer}" in
	[nN] | [nN][oO])
		echo ">>> Aborted by user."
		exit 1
		;;
	*) ;;
	esac
}

# =========================
# Sanity hints (non-fatal)
# =========================
echo ">>> Assumptions:"
echo "    - NVIDIA driver >= r580 (Blackwell)"
echo "    - CUDA Toolkit >= 13.1"
echo "    - Blackwell GPU (CC 10.x)"
echo

ask_continue "Proceed with environment setup?"

# =========================
# Check / Install conda
# =========================
if command -v conda >/dev/null 2>&1; then
	echo ">>> conda found: $(conda --version)"
	CONDA_BASE="$(conda info --base 2>/dev/null || true)"
	if [ -n "${CONDA_BASE}" ]; then
		CONDA_ACTIVATE="${CONDA_BASE}/bin/activate"
	else
		CONDA_BIN="$(command -v conda)"
		if [[ "${CONDA_BIN}" == /* ]]; then
			CONDA_ACTIVATE="$(dirname "${CONDA_BIN}")/activate"
		fi
	fi
	eval "$(conda shell.bash hook)"
elif [ -x "${MINICONDA_INSTALL_DIR}/bin/conda" ]; then
	echo ">>> conda found at ${MINICONDA_INSTALL_DIR}/bin/conda"
	CONDA_ACTIVATE="${MINICONDA_INSTALL_DIR}/bin/activate"
	eval "$("${MINICONDA_INSTALL_DIR}/bin/conda" shell.bash hook)"
elif [ -x /opt/conda/bin/conda ]; then
	echo ">>> conda found at /opt/conda/bin/conda"
	CONDA_ACTIVATE="/opt/conda/bin/activate"
	eval "$(/opt/conda/bin/conda shell.bash hook)"
else
	echo ">>> conda not found."
	ask_continue "Install Miniconda to ${MINICONDA_INSTALL_DIR}?"

	MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
	curl -fsSL "${MINICONDA_URL}" -o "${MINICONDA_INSTALLER}"
	bash "${MINICONDA_INSTALLER}" -b -p "${MINICONDA_INSTALL_DIR}"
	rm -f "${MINICONDA_INSTALLER}"

	# Activate conda for current session
	CONDA_ACTIVATE="${MINICONDA_INSTALL_DIR}/bin/activate"
	eval "$("${MINICONDA_INSTALL_DIR}/bin/conda" shell.bash hook)"

	# Initialize conda for future shells (both bash and zsh)
	"${MINICONDA_INSTALL_DIR}/bin/conda" init bash
	"${MINICONDA_INSTALL_DIR}/bin/conda" init zsh
	echo ">>> Miniconda installed at ${MINICONDA_INSTALL_DIR}"
	echo ">>> Please restart your shell or run: source ~/.bashrc (or ~/.zshrc)"
fi

# =========================
# Accept conda Terms of Service
# =========================
echo ">>> Accepting conda channel Terms of Service"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# =========================
# Create conda environment
# =========================
if conda env list | grep -q "^${ENV_NAME} "; then
	echo ">>> Found existing conda environment: ${ENV_NAME}"
	ask_continue "Reuse existing environment?"
else
	echo ">>> Will create conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
	ask_continue "Create new conda environment?"
	conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}" --override-channels -c conda-forge
fi

# Activate the environment
# Note: In non-interactive shells, we need to use conda run or source activate
if [ "${AUTO_YES}" = true ]; then
	# For non-interactive mode, set up the environment path directly
	CONDA_ENV_PATH=$(conda info --envs | grep "^${ENV_NAME} " | awk '{print $NF}')
	export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
	export CONDA_PREFIX="${CONDA_ENV_PATH}"
	echo ">>> Activated environment: ${ENV_NAME}"
else
	conda activate "${ENV_NAME}"
fi

if [ -z "${CONDA_ACTIVATE}" ]; then
	CONDA_BASE="$(conda info --base 2>/dev/null || true)"
	if [ -n "${CONDA_BASE}" ]; then
		CONDA_ACTIVATE="${CONDA_BASE}/bin/activate"
	fi
fi

# =========================
# Install CUDA Toolkit
# =========================
echo ">>> Installing CUDA Toolkit from nvidia channel"
ask_continue "Install CUDA Toolkit via conda?"
conda install -y nvidia::cuda

# =========================
# Core CUDA Python stack
# =========================
echo ">>> Installing CUDA Python stack (CUDA 13)"

python -m pip install "cupy-${CUDA_TAG}" cuda-tile numpy

# =========================
# PyTorch (architecture-specific)
# =========================
# Install PyTorch BEFORE other CUDA Python packages to prevent
# accelerate from pulling the wrong torch build from default PyPI.
echo ">>> Installing PyTorch (Blackwell, CUDA 13.x)"
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# =========================
# CUDA Environment Variables
# =========================
# echo ">>> Configuring CUDA environment variables..."

# CONDA_ENV_PATH=$(conda info --envs | grep "^${ENV_NAME} " | awk '{print $NF}')
# if [ -n "${CONDA_ENV_PATH}" ]; then
# 	mkdir -p "${CONDA_ENV_PATH}/etc/conda/activate.d"
# 	mkdir -p "${CONDA_ENV_PATH}/etc/conda/deactivate.d"

# 	# Get the project root directory (parent of utils/)
# 	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 	PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 	# Create activation script
# 	cat >"${CONDA_ENV_PATH}/etc/conda/activate.d/cutile_env.sh" <<EOF
# #!/bin/bash
# # CUDA_PATH for CuPy to find CUDA headers
# export CUDA_PATH=\${CONDA_PREFIX}/targets/x86_64-linux
# EOF

# 	# Create deactivation script
# 	cat >"${CONDA_ENV_PATH}/etc/conda/deactivate.d/cutile_env.sh" <<'EOF'
# #!/bin/bash
# unset CUDA_PATH
# EOF
# 	echo "    CUDA_PATH configured for CuPy."
# fi

# =========================
# Optional but recommended
# =========================
echo ">>> Installing optional tooling"

# NVML access (driver introspection, useful for debugging)
python -m pip install pynvml

# NumPy (used by almost all examples)
python -m pip install numpy

# =========================
# HuggingFace & ML Tools (for hw1-asr and beyond)
# =========================
echo ">>> Installing HuggingFace ecosystem and ML tools"

# HuggingFace ecosystem
python -m pip install transformers datasets huggingface_hub accelerate safetensors

# Streamlit for web apps
python -m pip install streamlit

# Audio processing (for ASR tasks)
python -m pip install soundfile librosa

# =========================
# Fix cuda-bindings version conflict
# =========================
# PyTorch pins cuda-bindings to its bundled version, but cuda-python/cuda-tile
# need ~=13.1.1. Force 13.1.x last so it isn't overwritten by transitive deps.
# echo ">>> Fixing cuda-bindings version for cuTile compatibility..."
# python -m pip install "cuda-bindings~=13.1.1" "cuda-python~=13.1.1" --force-reinstall --quiet

# =========================
# Freeze snapshot
# =========================
echo ">>> Writing lock snapshot (requirements.lock)"
conda list --export >requirements.lock

# =========================
# Done
# =========================
echo
echo "============================================="
echo " MLS Python environment is ready."
echo " (Machine Learning Systems - cutile + hw1)"
echo "============================================="
echo
echo "If conda isn't on PATH, run:"
echo "  source ${CONDA_ACTIVATE}"
echo "  conda activate ${ENV_NAME}"
echo
echo "Installed key packages:"
echo "  CUDA/cuTile stack:"
echo "    - nvidia::cuda (via conda)"
echo "    - cupy-${CUDA_TAG}"
echo "    - cuda-python"
echo "    - cuda-tile"
echo
echo "  HuggingFace & ML:"
echo "    - transformers, datasets, huggingface_hub, safetensors"
echo "    - torch, torchvision (cu130 for Blackwell)"
echo "    - streamlit"
echo "    - soundfile, librosa"
echo
echo "Architecture: Blackwell (native support)"
echo
echo "NOTE: If you installed Miniconda now, restart your shell if needed."
echo
