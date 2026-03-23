import torch

# from lib import utils

def get_random_phase(n, device="cuda"):
    """
    生成长度为n/2的随机复位相位 
    """ 
    assert n % 2 == 0, "n must be even"
    angles = torch.rand(n // 2, device=device) * 2 * torch.pi
    phase = torch.exp(1j * angles)
    return phase

def matmul_rfft(x, phase):
    """
    """ 
    n = x.shape[-1] 
    x = x.contiguous()
    x_complex = torch.view_as_complex(x.view(*x.shape[:-1], n // 2, 2)) 
    x_complex = x_complex * phase
    x_fft = torch.fft.fft(x_complex, norm="ortho") 
    x_out = torch.view_as_real(x_fft).view(*x.shape[:-1], n)
    return x_out 

def RFFT_H(H, SU, phase):
    """
    使用 RFFT 对海森矩阵 H 进行不相干处理。
    替换原有的 RHT_H 函数。
    
    参数:
        H: 海森矩阵 (n, n)
        SU: 输入缩放因子 (n,) 或 (1, n)
        phase: 预先生成的随机复数相位 (n//2,)
    """
    # 第一步：对 H * SU 的行进行 RFFT 变换
    step1 = matmul_rfft(H * SU, phase)
    
    # 第二步：转置后，再次乘以 SU，并对新的行（原矩阵的列）进行 RFFT 变换
    # 这与 utils.matmul_hadUt(utils.matmul_hadUt(H * SU).T * SU) 的逻辑完全一致
    step2 = matmul_rfft(step1.T * SU, phase)
    
    return step2

def RFFT_W(W, SU, SV, phase_out, phase_in):
    """
    使用 RFFT 对权重矩阵 W 进行不相干处理。
    
    参数:
        W: 权重矩阵 (m, n)
        SU: 输入缩放因子 (n,) 或 (1, n)
        SV: 输出缩放因子 (m,) 或 (m, 1)
        phase_out: 输出维度的随机复数相位 (m//2,)
        phase_in: 输入维度的随机复数相位 (n//2,)
    """
    # 对 W 的行进行 RFFT 变换
    step1 = matmul_rfft(W.T * SV, phase_out)
    
    # 转置后，对列进行 RFFT 变换
    step2 = matmul_rfft(step1.T * SU, phase_in)
    
    return step2


if __name__ == "__main__":
# 假设一个线性层 W 的形状为 (4096, 8192)
    m = 4096 
    n = 4096
    W = torch.randn(m, n, device='cuda')
    SU = torch.ones(n, device='cuda') # 根据原代码逻辑，SU 对应外层 n 维度
    SV = torch.ones(m, device='cuda') # SV 对应内层 m 维度

    # 1. 为输入和输出维度分别生成固定的随机相位
    phase_out = get_random_phase(m, device=W.device) # 长度 2048
    phase_in = get_random_phase(n, device=W.device)  # 长度 4096

    # 2. 处理权重矩阵
    W_processed = RFFT_W(W, SU, SV, phase_out, phase_in)
    print("Original W shape:", W.shape)
    print("Processed W shape:", W_processed.shape)
    