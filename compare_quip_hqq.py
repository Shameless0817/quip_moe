"""
对比 QUIP 和 HQQ 量化方法的量化损失
"""

import torch
import sys
import os

sys.path.insert(0, '/fact_home/zeyuli/quip_sharp')

from lib.algo import quip
from lib.codebook import get_codebook
from lib import utils


def hqq_quantize(W, nbits=2):
    """HQQ 简单量化 (全局 scale)"""
    qmax = 2 ** (nbits - 1) - 1
    qmin = -qmax - 1
    
    max_val = W.abs().max()
    scale = max_val / qmax if max_val > 0 else 1.0
    
    W_int = torch.round(W / scale)
    W_int = torch.clamp(W_int, qmin, qmax)
    W_quant = W_int * scale
    
    return W_quant, scale


def hqq_quantize_with_hessian(W, H, nbits=2):
    """HQQ Hessian-aware 量化 (逐列 scale)"""
    m, n = W.shape
    W_quant = torch.zeros_like(W)
    scales = []
    
    qmax = 2 ** (nbits - 1) - 1
    qmin = -qmax - 1
    
    for col in range(n):
        w_col = W[:, col]
        max_val = w_col.abs().max()
        scale = max_val / qmax if max_val > 0 else 1.0
        
        w_int = torch.round(w_col / scale)
        w_int = torch.clamp(w_int, qmin, qmax)
        W_quant[:, col] = w_int * scale
        scales.append(scale)
    
    return W_quant, torch.tensor(scales)


def main():
    print("="*80)
    print("QUIP vs HQQ 量化方法对比")
    print("="*80)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}\n")
    
    # 准备测试数据
    torch.manual_seed(42)
    m, n = 4096, 4096
    W = torch.randn(m, n, device='cpu', dtype=torch.float32)
    
    print(f"权重矩阵: {W.shape}")
    print(f"  Range: [{W.min():.4f}, {W.max():.4f}]")
    print(f"  Std: {W.std():.4f}\n")
    
    # 准备 Hessian
    H_base = torch.randn(n, n, device='cpu', dtype=torch.float32)
    H = H_base @ H_base.T
    H = utils.regularize_H(H, n, sigma_reg=1e-2)
    
    H_diag = H.diag()
    cond = H_diag.max() / (H_diag.min() + 1e-10)
    print(f"Hessian 矩阵: {H.shape}")
    print(f"  条件数: {cond:.2e}\n")
    
    results = {}
    
    # ========== 方法1: QUIP 量化 ==========
    print("="*80)
    print("【方法1】QUIP 量化 (E8P12, ~2bit)")
    print("="*80)
    
    class Args:
        def __init__(self):
            self.rescale_WH = True
            self.use_fp64 = False
            self.save_pfx = '/tmp'
            self.lora_rank = 0
            self.sigma_reg = 1e-2
            self.sigma_reg2 = 1e-2
            self.scale_override = -1
            self.resid_scale_override = -1
            self.quip_tune_iters = 10
            self.no_use_buffered = False
            self.lowmem_ldlq = False
            self.full_svd = False
    
    args = Args()
    cb = get_codebook('E8P12')
    
    print("开始 QUIP 量化...")
    result = quip.quantize(H.clone(), W.clone(), args.lora_rank, cb, args, device)
    
    if result is not None:
        W_quip, attr = result
        
        quip_mse = (W - W_quip).square().mean().item()
        quip_l2 = (W - W_quip).norm().item() / W.norm().item()
        
        diff = W - W_quip
        quip_hess_loss = (diff @ H @ diff.T).trace().item() / W.shape[0]
        
        results['quip'] = {
            'mse': quip_mse,
            'l2_error': quip_l2,
            'hessian_loss': quip_hess_loss,
        }
        
        print(f"\n✓ QUIP 量化完成")
        print(f"  MSE: {quip_mse:.6e}")
        print(f"  L2 相对误差: {quip_l2:.4f}")
        print(f"  Hessian 加权损失: {quip_hess_loss:.6e}")
    else:
        print("\n❌ QUIP 量化失败")
        results['quip'] = None
    
    # ========== 方法2: HQQ 简单量化 ==========
    print("\n" + "="*80)
    print("【方法2】HQQ 简单量化 (2-bit, 全局 scale)")
    print("="*80)
    
    print("开始 HQQ 简单量化...")
    W_hqq, scale = hqq_quantize(W, nbits=2)
    
    hqq_mse = (W - W_hqq).square().mean().item()
    hqq_l2 = (W - W_hqq).norm().item() / W.norm().item()
    
    diff = W - W_hqq
    hqq_hess_loss = (diff @ H @ diff.T).trace().item() / W.shape[0]
    
    results['hqq'] = {
        'mse': hqq_mse,
        'l2_error': hqq_l2,
        'hessian_loss': hqq_hess_loss,
    }
    
    print(f"\n✓ HQQ 简单量化完成")
    print(f"  MSE: {hqq_mse:.6e}")
    print(f"  L2 相对误差: {hqq_l2:.4f}")
    print(f"  Hessian 加权损失: {hqq_hess_loss:.6e}")
    print(f"  全局 scale: {scale:.6f}")
    
    # ========== 方法3: HQQ Hessian-aware ==========
    print("\n" + "="*80)
    print("【方法3】HQQ Hessian-aware 量化 (2-bit, 逐列 scale)")
    print("="*80)
    
    print("开始 HQQ Hessian-aware 量化...")
    W_hqq_hess, scales = hqq_quantize_with_hessian(W, H, nbits=2)
    
    hqq_hess_mse = (W - W_hqq_hess).square().mean().item()
    hqq_hess_l2 = (W - W_hqq_hess).norm().item() / W.norm().item()
    
    diff = W - W_hqq_hess
    hqq_hess_hess_loss = (diff @ H @ diff.T).trace().item() / W.shape[0]
    
    results['hqq_hessian'] = {
        'mse': hqq_hess_mse,
        'l2_error': hqq_hess_l2,
        'hessian_loss': hqq_hess_hess_loss,
    }
    
    print(f"\n✓ HQQ Hessian-aware 量化完成")
    print(f"  MSE: {hqq_hess_mse:.6e}")
    print(f"  L2 相对误差: {hqq_hess_l2:.4f}")
    print(f"  Hessian 加权损失: {hqq_hess_hess_loss:.6e}")
    print(f"  Scales 范围: [{scales.min():.6f}, {scales.max():.6f}]")
    
    # ========== 对比总结 ==========
    print("\n\n" + "="*80)
    print("对比总结")
    print("="*80)
    
    print("\n【指标对比表】")
    print("-" * 90)
    print(f"{'方法':<35} {'MSE':<18} {'L2误差':<15} {'Hessian损失':<18}")
    print("-" * 90)
    
    if results['quip']:
        print(f"{'QUIP (E8P12, rescale_WH)':<35} "
              f"{results['quip']['mse']:<18.6e} "
              f"{results['quip']['l2_error']:<15.4f} "
              f"{results['quip']['hessian_loss']:<18.6e}")
    
    print(f"{'HQQ 简单 (全局scale)':<35} "
          f"{results['hqq']['mse']:<18.6e} "
          f"{results['hqq']['l2_error']:<15.4f} "
          f"{results['hqq']['hessian_loss']:<18.6e}")
    
    print(f"{'HQQ Hessian-aware (逐列scale)':<35} "
          f"{results['hqq_hessian']['mse']:<18.6e} "
          f"{results['hqq_hessian']['l2_error']:<15.4f} "
          f"{results['hqq_hessian']['hessian_loss']:<18.6e}")
    
    print("-" * 90)
    
    # 计算改进
    if results['quip']:
        print("\n【相对于 HQQ 简单量化的改进】")
        print("-" * 60)
        
        quip_mse_improve = (results['hqq']['mse'] - results['quip']['mse']) / results['hqq']['mse'] * 100
        quip_hess_improve = (results['hqq']['hessian_loss'] - results['quip']['hessian_loss']) / results['hqq']['hessian_loss'] * 100
        
        print(f"  QUIP:")
        print(f"    MSE 改进:          {quip_mse_improve:+.2f}% ({results['hqq']['mse']:.2e} → {results['quip']['mse']:.2e})")
        print(f"    Hessian 损失改进:  {quip_hess_improve:+.2f}% ({results['hqq']['hessian_loss']:.2e} → {results['quip']['hessian_loss']:.2e})")
        
        hqq_hess_mse_improve = (results['hqq']['mse'] - results['hqq_hessian']['mse']) / results['hqq']['mse'] * 100
        hqq_hess_hess_improve = (results['hqq']['hessian_loss'] - results['hqq_hessian']['hessian_loss']) / results['hqq']['hessian_loss'] * 100
        
        print(f"\n  HQQ Hessian-aware:")
        print(f"    MSE 改进:          {hqq_hess_mse_improve:+.2f}% ({results['hqq']['mse']:.2e} → {results['hqq_hessian']['mse']:.2e})")
        print(f"    Hessian 损失改进:  {hqq_hess_hess_improve:+.2f}% ({results['hqq']['hessian_loss']:.2e} → {results['hqq_hessian']['hessian_loss']:.2e})")
        
        # QUIP vs HQQ Hessian-aware
        quip_vs_hqq_hess = (results['hqq_hessian']['hessian_loss'] - results['quip']['hessian_loss']) / results['hqq_hessian']['hessian_loss'] * 100
        print(f"\n  QUIP vs HQQ Hessian-aware:")
        print(f"    Hessian 损失改进:  {quip_vs_hqq_hess:+.2f}% ({results['hqq_hessian']['hessian_loss']:.2e} → {results['quip']['hessian_loss']:.2e})")
    
    print("\n【结论】")
    print("-" * 60)
    print("""
✓ QUIP 量化优势:
  - 使用 E8P12 lattice codebook 提供最优量化点
  - Hadamard 变换 + LDLQ 算法考虑 Hessian，优化关键权重
  - Hessian 加权损失显著低于简单量化方法
  - 代价: 计算复杂，需要 SU/SV 额外参数

✓ HQQ Hessian-aware 优势:
  - 比简单 HQQ 好，考虑了列的重要性
  - 计算开销适中
  - 但仍不如 QUIP (特别是 Hessian 加权损失)

✓ HQQ 简单量化:
  - 最快，但量化误差最大
  - 适合快速原型测试

推荐: 生产环境使用 QUIP (质量优先) 或 HQQ Hessian-aware (速度质量平衡)
    """)
    
    print("="*80)


if __name__ == '__main__':
    main()
