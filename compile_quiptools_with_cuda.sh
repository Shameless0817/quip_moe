#!/bin/bash

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Compiling quiptools_cuda with CUDA"
echo "=========================================="
echo ""

# Step 1: 查找系统中的 nvcc
echo "[1] Searching for nvcc..."
NVCC_LOCATIONS=(
    "/usr/local/cuda/bin/nvcc"
    "/usr/local/cuda-12.4/bin/nvcc"
    "/usr/local/cuda-12.1/bin/nvcc"
    "/usr/local/cuda-11.8/bin/nvcc"
)

FOUND_NVCC=""
for loc in "${NVCC_LOCATIONS[@]}"; do
    if [ -f "$loc" ]; then
        FOUND_NVCC="$loc"
        echo "  Found nvcc at: $FOUND_NVCC"
        break
    fi
done

if [ -z "$FOUND_NVCC" ]; then
    echo "  Searching in filesystem..."
    FOUND_NVCC=$(find /usr/local -name nvcc -type f 2>/dev/null | head -n 1)
fi

if [ -z "$FOUND_NVCC" ]; then
    echo ""
    echo "ERROR: nvcc not found!"
    echo ""
    echo "Please install CUDA Toolkit:"
    echo "  Option 1: conda install -c nvidia cuda-toolkit=12.1"
    echo "  Option 2: Download from https://developer.nvidia.com/cuda-downloads"
    echo ""
    exit 1
fi

# 设置 CUDA_HOME
CUDA_HOME=$(dirname $(dirname "$FOUND_NVCC"))
echo "  CUDA_HOME: $CUDA_HOME"
echo ""

# Step 2: 验证 nvcc 版本
echo "[2] NVCC version:"
"$FOUND_NVCC" --version | grep "release"
echo ""

# Step 3: 检查 PyTorch
echo "[3] PyTorch version:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.version.cuda}'); print(f'  Available: {torch.cuda.is_available()}')"
echo ""

# Step 4: 设置环境变量
echo "[4] Setting environment variables..."
export CUDA_HOME="$CUDA_HOME"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

echo "  CUDA_HOME=$CUDA_HOME"
echo "  CUDA_PATH=$CUDA_PATH"
echo "  PATH includes: $CUDA_HOME/bin"
echo ""

# Step 5: 验证 nvcc 可访问
echo "[5] Verifying nvcc is accessible..."
which nvcc || echo "  Warning: nvcc not in PATH, but CUDA_HOME is set"
echo ""

# Step 6: 进入 quiptools 目录
cd /fact_home/zeyuli/quip_sharp/quiptools || {
    echo "ERROR: Cannot enter quiptools directory!"
    exit 1
}
echo "[6] Working in: $(pwd)"
echo ""

# Step 7: 清理
echo "[7] Cleaning old builds..."
pip uninstall -y quiptools_cuda 2>/dev/null || true
rm -rf build/ dist/ *.egg-info __pycache__ *.so
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "  Cleaned!"
echo ""

# Step 8: 编译
echo "[8] Compiling (this may take 2-3 minutes)..."
echo "  Using CUDA_HOME: $CUDA_HOME"
echo ""

# 使用 pip install 并传递环境变量
CUDA_HOME="$CUDA_HOME" \
CUDA_PATH="$CUDA_PATH" \
PATH="$CUDA_HOME/bin:$PATH" \
LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}" \
pip install -e . --no-build-isolation 2>&1 | tee /tmp/quiptools_compile.log

BUILD_EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "SUCCESS!"
    echo "=========================================="
    echo ""
    echo "[9] Testing import..."
    python -c "import quiptools_cuda; print('✓ quiptools_cuda loaded successfully!')"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "All done! ✓"
        echo ""
        echo "To make CUDA settings permanent, add to ~/.bashrc or ~/.zshrc:"
        echo "  export CUDA_HOME=$CUDA_HOME"
        echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
        echo "  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
    else
        echo ""
        echo "WARNING: Compilation succeeded but import failed!"
    fi
else
    echo "=========================================="
    echo "COMPILATION FAILED!"
    echo "=========================================="
    echo ""
    echo "Check the log: /tmp/quiptools_compile.log"
    echo ""
    echo "Common fixes:"
    echo "  1. Check GCC version: gcc --version (should be compatible with CUDA)"
    echo "  2. Reinstall PyTorch with CUDA: pip install torch --force-reinstall"
    echo "  3. Try different CUDA version: conda install cuda-toolkit=11.8"
    exit 1
fi
