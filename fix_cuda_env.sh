#!/bin/bash

echo "=========================================="
echo "CUDA Environment Diagnostic and Fix Script"
echo "=========================================="
echo ""

# Step 1: 检查 nvidia-smi
echo "[1] Checking GPU and CUDA runtime version..."
nvidia-smi 2>/dev/null | head -n 5
if [ $? -ne 0 ]; then
    echo "ERROR: nvidia-smi not found. Please check if NVIDIA driver is installed."
    exit 1
fi
echo ""

# Step 2: 查找系统中的 nvcc
echo "[2] Searching for nvcc in common locations..."
NVCC_PATHS=(
    "/usr/local/cuda/bin/nvcc"
    "/usr/local/cuda-12.4/bin/nvcc"
    "/usr/local/cuda-12.1/bin/nvcc"
    "/usr/local/cuda-11.8/bin/nvcc"
    "/opt/cuda/bin/nvcc"
)

FOUND_NVCC=""
for nvcc_path in "${NVCC_PATHS[@]}"; do
    if [ -f "$nvcc_path" ]; then
        echo "  Found: $nvcc_path"
        FOUND_NVCC=$nvcc_path
        break
    fi
done

if [ -z "$FOUND_NVCC" ]; then
    echo "  Searching in file system..."
    FOUND_NVCC=$(find /usr/local -name nvcc 2>/dev/null | head -n 1)
fi

if [ -z "$FOUND_NVCC" ]; then
    echo "ERROR: nvcc not found in system!"
    echo "Please install CUDA Toolkit:"
    echo "  conda install -c nvidia cuda-toolkit"
    echo "OR download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo "  Using nvcc at: $FOUND_NVCC"
CUDA_HOME=$(dirname $(dirname $FOUND_NVCC))
echo "  CUDA_HOME: $CUDA_HOME"
echo ""

# Step 3: 验证 nvcc 版本
echo "[3] Checking nvcc version..."
$FOUND_NVCC --version
echo ""

# Step 4: 检查 PyTorch CUDA 版本
echo "[4] Checking PyTorch CUDA version..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'PyTorch CUDA: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: Could not check PyTorch version"
fi
echo ""

# Step 5: 设置环境变量
echo "[5] Setting up environment variables..."
export CUDA_HOME=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "  CUDA_HOME=$CUDA_HOME"
echo "  PATH includes: $CUDA_HOME/bin"
echo "  LD_LIBRARY_PATH includes: $CUDA_HOME/lib64"
echo ""

# Step 6: 验证 nvcc 现在可用
echo "[6] Verifying nvcc is now accessible..."
which nvcc
nvcc --version | grep "release"
echo ""

# Step 7: 编译 quiptools_cuda
echo "[7] Compiling quiptools_cuda..."
cd /fact_home/zeyuli/quip_sharp/quiptools

echo "  Uninstalling old version..."
pip uninstall -y quiptools_cuda 2>/dev/null

echo "  Installing quiptools_cuda..."
pip install -e . 2>&1 | tee /tmp/quiptools_install.log

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "SUCCESS! quiptools_cuda compiled successfully!"
    echo "=========================================="
    echo ""
    echo "To make these settings permanent, add to your ~/.bashrc:"
    echo "  export CUDA_HOME=$CUDA_HOME"
    echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
    echo "  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "FAILED! Check the log at /tmp/quiptools_install.log"
    echo "=========================================="
    echo ""
    echo "Common issues:"
    echo "  1. CUDA Toolkit version mismatch with PyTorch"
    echo "  2. Missing development headers (gcc, g++)"
    echo "  3. Insufficient permissions"
    echo ""
    exit 1
fi
