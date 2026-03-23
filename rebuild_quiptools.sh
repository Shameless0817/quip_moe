#!/bin/bash

echo "=========================================="
echo "Rebuilding quiptools_cuda"
echo "=========================================="
echo ""

# Step 1: 检查当前 PyTorch 版本
echo "[1] Checking current PyTorch version..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}')"
if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch not found!"
    exit 1
fi
echo ""

# Step 2: 设置 CUDA 环境
echo "[2] Setting up CUDA environment..."
FOUND_NVCC=$(find /usr/local -name nvcc 2>/dev/null | head -n 1)
if [ -z "$FOUND_NVCC" ]; then
    FOUND_NVCC="/usr/local/cuda/bin/nvcc"
fi

if [ -f "$FOUND_NVCC" ]; then
    CUDA_HOME=$(dirname $(dirname $FOUND_NVCC))
    export CUDA_HOME=$CUDA_HOME
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    echo "  CUDA_HOME: $CUDA_HOME"
    which nvcc
else
    echo "WARNING: nvcc not found, continuing anyway..."
fi
echo ""

# Step 3: 进入 quiptools 目录
cd /fact_home/zeyuli/quip_sharp/quiptools || { echo "ERROR: quiptools directory not found!"; exit 1; }
echo "[3] Working directory: $(pwd)"
echo ""

# Step 4: 完全清理旧的编译产物
echo "[4] Cleaning old build artifacts..."
pip uninstall -y quiptools_cuda 2>/dev/null
rm -rf build/ dist/ *.egg-info __pycache__
rm -f quiptools_cuda*.so
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -delete
echo "  Cleaned!"
echo ""

# Step 5: 清理 pip 缓存
echo "[5] Cleaning pip cache..."
pip cache purge 2>/dev/null || pip cache remove quiptools_cuda 2>/dev/null || true
echo ""

# Step 6: 重新编译并安装
echo "[6] Rebuilding quiptools_cuda (this may take a few minutes)..."
echo "  Please wait..."
pip install -e . --no-cache-dir --force-reinstall 2>&1 | tee /tmp/quiptools_rebuild.log

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "SUCCESS!"
    echo "=========================================="
    echo ""
    echo "[7] Verifying installation..."
    python -c "import quiptools_cuda; print('✓ quiptools_cuda imported successfully!')" 2>&1
    if [ $? -eq 0 ]; then
        echo ""
        echo "All done! quiptools_cuda is ready to use."
    else
        echo ""
        echo "WARNING: Installation succeeded but import failed."
        echo "Check the error above."
    fi
else
    echo ""
    echo "=========================================="
    echo "FAILED!"
    echo "=========================================="
    echo ""
    echo "Check the log: /tmp/quiptools_rebuild.log"
    echo ""
    echo "Common issues:"
    echo "  1. CUDA Toolkit not installed or wrong version"
    echo "  2. GCC/G++ version incompatibility"
    echo "  3. PyTorch installed without CUDA support"
    echo ""
    echo "Try:"
    echo "  pip install torch --force-reinstall"
    echo "  conda install -c nvidia cuda-toolkit"
    exit 1
fi
