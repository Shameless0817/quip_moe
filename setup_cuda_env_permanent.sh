#!/bin/bash

# 这个脚本将 CUDA 环境变量永久添加到你的 shell 配置文件中

echo "Setting up permanent CUDA environment..."

# 查找 nvcc
FOUND_NVCC=$(find /usr/local -name nvcc 2>/dev/null | head -n 1)
if [ -z "$FOUND_NVCC" ]; then
    FOUND_NVCC="/usr/local/cuda/bin/nvcc"
fi

if [ ! -f "$FOUND_NVCC" ]; then
    echo "ERROR: nvcc not found!"
    exit 1
fi

CUDA_HOME=$(dirname $(dirname $FOUND_NVCC))
echo "Detected CUDA_HOME: $CUDA_HOME"

# 检测使用的 shell
SHELL_CONFIG=""
if [ -n "$ZSH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
else
    echo "Unknown shell. Please manually add the following to your shell config:"
    echo ""
    echo "export CUDA_HOME=$CUDA_HOME"
    echo "export PATH=\$CUDA_HOME/bin:\$PATH"
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
    exit 1
fi

echo "Adding to $SHELL_CONFIG..."

# 检查是否已经添加
if grep -q "CUDA_HOME.*$CUDA_HOME" "$SHELL_CONFIG" 2>/dev/null; then
    echo "CUDA settings already present in $SHELL_CONFIG"
else
    echo "" >> "$SHELL_CONFIG"
    echo "# CUDA Environment (added by setup_cuda_env_permanent.sh)" >> "$SHELL_CONFIG"
    echo "export CUDA_HOME=$CUDA_HOME" >> "$SHELL_CONFIG"
    echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> "$SHELL_CONFIG"
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> "$SHELL_CONFIG"
    echo "" >> "$SHELL_CONFIG"
    echo "Added CUDA settings to $SHELL_CONFIG"
fi

echo ""
echo "Done! Please run: source $SHELL_CONFIG"
echo "Or restart your terminal."
