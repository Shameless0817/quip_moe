import time

import quiptools_cuda
import torch
import torch.nn as nn

from lib import codebook
from lib.utils import clean, dtype_from_str, get_hadK


class QuantizedLinearWithlora(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        codesz,
        packsz,
        pack_out,
        idx_dtype,
        codebook_version,
        rank=-1,
        lora_alpha=0.8,
        lora_trainable=False,
        rescale_WH=False,
        bias=False,
        resid_scale_override=-1,
        train_mode=False,
        grad_ckpt=False,
    ):
        super().__init__()
        # assert rank == 0 # 7/22/2024 removed support for low rank correction
        # 恢复低秩修正支持
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_trainable = lora_trainable
        self.rescale_WH = rescale_WH
        self.resid_scale_override = resid_scale_override

        self.has_bias = bias
        if self.has_bias:
            self.register_buffer('bias', torch.ones(out_features))

        if self.rank > 0:
            if lora_trainable:
                self.lora_A = nn.Parameter(torch.zeros(out_features, rank))
                self.lora_B = nn.Parameter(torch.zeros(rank, in_features))
                ## 暂时还不支持QAT训练lora，尽情期待后续更新
            else:
                self.register_buffer('lora_A', torch.zeros(out_features, rank))
                self.register_buffer('lora_B', torch.zeros(rank, in_features))
        else:
            self.lora_A = None
            self.lora_B = None

        if self.rescale_WH:
            self.register_buffer("scaleWH", torch.ones(in_features))
        else:
            self.scaleWH = None

        # direction we pack in, the code dimension is always in the in dimension
        if pack_out:
            self.register_buffer(
                "Qidxs",
                torch.zeros(int(out_features / packsz),
                            int(in_features / codesz),
                            dtype=dtype_from_str(idx_dtype)))
        else:
            self.register_buffer(
                "Qidxs",
                torch.zeros(out_features,
                            int(in_features / (codesz * packsz)),
                            dtype=dtype_from_str(idx_dtype)))

        self.register_buffer("codebook_id", torch.tensor(7))
        self.register_buffer("SU", torch.ones(in_features,
                                              dtype=torch.float16))
        self.register_buffer("SV", torch.ones(out_features,
                                              dtype=torch.float16))
        self.register_buffer("Wscale", torch.ones(()))

        self.built_codebook_class = False
        self.built_graph = False
        self.codebook_version = codebook_version

        had_left_T, K_left = get_hadK(in_features)
        if had_left_T is not None:
            had_left_T = had_left_T.T.contiguous()
        self.register_buffer('had_left_T', had_left_T, persistent=False)
        
        had_right, K_right = get_hadK(out_features)
        self.register_buffer('had_right', had_right, persistent=False)
        
        self.K_left = K_left
        self.K_right = K_right
        self.packed = (packsz != 1)
        self.train_mode = train_mode
        self.grad_ckpt = grad_ckpt

    def decompress_weights(self):
        """解压缩量化权重为完整的浮点权重矩阵"""
        # 确定目标设备 - 解压操作需要 CUDA
        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if target_device.type == 'cpu':
            raise RuntimeError("decompress_weights requires CUDA, but CUDA is not available")
        
        # 如果还没有初始化 codebook class，先初始化
        if not self.built_codebook_class:
            # 确保使用 CUDA 设备初始化 codebook
            self.codebook_class = codebook.get_quantized_class(
                self.codebook_id.item()
            )(target_device)
            
            if self.codebook_class.codebook.version != self.codebook_version:
                raise Exception(
                    f"Saved weights version ({self.codebook_version}) does not match the "
                    f"codebook version ({self.codebook_class.codebook.version}). "
                    "Please download the latest weights from https://huggingface.co/relaxml")

            # 解包 Qidxs - 先移到 CPU 进行解包，然后移回目标设备
            original_device = self.Qidxs.device
            self.Qidxs = self.Qidxs.cpu()
            split_qidxs = self.codebook_class.maybe_unpack_idxs(self.Qidxs)
            self.Qidxs_list = []
            for i in range(len(split_qidxs)):
                # 将解包后的索引移到目标设备（CUDA）
                self.register_buffer(f'Qidxs_{i}', split_qidxs[i].to(target_device))
                exec(f'self.Qidxs_list.append(self.Qidxs_{i})')
            del self.Qidxs

            # fuse Wscale into SV
            self.SV *= self.Wscale
            
            self.built_codebook_class = True
        
        # 现在可以解压权重了
        m, n = self.out_features, self.in_features
        
        # Qidxs_list 应该已经在正确的设备上且是正确的类型（int64）
        qidxs = self.Qidxs_list[0]
        if qidxs.device != target_device:
            qidxs = qidxs.to(target_device)
        
        # 验证类型是否正确
        if qidxs.dtype != torch.int64:
            raise RuntimeError(f"Qidxs 应该是 int64 类型，但得到的是 {qidxs.dtype}。"
                             "这通常表示模块创建时使用了错误的 idx_dtype。")
            
        grid = self.codebook_class.codebook.grid_packed_abs
        if grid.device != target_device:
            grid = grid.to(target_device)
        
        # 验证 grid 类型
        if grid.dtype != torch.int32:
            raise RuntimeError(f"grid_packed_abs 应该是 int32 类型，但得到的是 {grid.dtype}")
        
        # 调用解压函数
        W_decompressed = torch.ops.quip_lib.decompress_packed_e8p(
            qidxs.view(m // 16, n // 64, 8, 4),
            grid,
            m, n
        )
        
        # 应用 scale 因子
        W_decompressed = W_decompressed.float() / self.codebook_class.scale
        
        # 应用 Hadamard 变换和缩放
        if self.had_left_T is not None and self.had_right is not None:
            from lib.utils.matmul_had import matmul_hadU_cuda
            had_left = self.had_left_T.T.contiguous().to(target_device) if self.had_left_T is not None else None
            had_right = self.had_right.to(target_device) if self.had_right is not None else None
            
            W_decompressed = matmul_hadU_cuda(
                matmul_hadU_cuda(W_decompressed, had_left, self.K_left).T,
                had_right,
                self.K_right
            )
        
        # 应用 SU 和 SV 缩放
        SU = self.SU.to(target_device)
        SV = self.SV.to(target_device)
        W_decompressed = W_decompressed * SV.unsqueeze(1) * SU.unsqueeze(0)
        
        return W_decompressed.to(torch.float16)
    
    def initialize_lora_from_decomposition(self, delta_W):
        """
        从权重增量矩阵初始化LoRA
        
        Args:
            delta_W: 权重增量矩阵 (out_features, in_features)
        """
        if self.rank <= 0:
            raise ValueError("rank必须大于0才能初始化LoRA")
        
        # SVD分解
        U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
        
        # 保留前rank个奇异值
        k = min(self.rank, len(S))
        
        # B = U[:, :k] @ diag(sqrt(S[:k]))
        # A = diag(sqrt(S[:k])) @ Vh[:k, :]
        sqrt_S = torch.sqrt(S[:k])
        
        if self.lora_trainable:
            self.lora_B.data = (U[:, :k] * sqrt_S.unsqueeze(0))
            self.lora_A.data = (sqrt_S.unsqueeze(1) * Vh[:k, :])
        else:
            self.lora_B = (U[:, :k] * sqrt_S.unsqueeze(0))
            self.lora_A = (sqrt_S.unsqueeze(1) * Vh[:k, :])
        
        print(f"LoRA已从SVD初始化，使用了 {k}/{len(S)} 个奇异值")

    def merge_lora_to_weights(self):
        """
        将LoRA权重合并回去（用于推理优化）
        
        Returns:
            合并后的权重增量: B @ A * alpha
        """
        if self.lora_A is None or self.lora_B is None:
            return None
        
        # 计算LoRA增量: ΔW = B @ A * alpha
        with torch.no_grad():
            delta_W = (self.lora_B @ self.lora_A) * self.lora_alpha
        
        return delta_W
    
    def enable_lora_training(self):
        """启用LoRA训练"""
        if self.lora_A is not None and isinstance(self.lora_A, nn.Parameter):
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True
            self.lora_trainable = True

    def disable_lora_training(self):
        """禁用LoRA训练"""
        if self.lora_A is not None and isinstance(self.lora_A, nn.Parameter):
            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False
            self.lora_trainable = False
    
    def get_lora_parameters(self):
        """返回LoRA参数（用于单独优化）"""
        if self.lora_A is not None and self.lora_B is not None:
            if isinstance(self.lora_A, nn.Parameter):
                return [self.lora_A, self.lora_B]
        return []

    def print_lora_info(self):
        """打印LoRA信息"""
        if self.lora_A is None or self.lora_B is None:
            print("该层没有LoRA")
            return
        
        lora_params = self.lora_A.numel() + self.lora_B.numel()
        base_params = self.in_features * self.out_features
        
        print(f"LoRA配置:")
        print(f"  秩 (rank): {self.rank}")
        print(f"  Alpha: {self.lora_alpha}")
        print(f"  可训练: {self.lora_trainable}")
        print(f"  LoRA参数量: {lora_params:,}")
        print(f"  基础参数量: {base_params:,}")
        print(f"  参数比例: {100 * lora_params / base_params:.4f}%")
        
        if isinstance(self.lora_A, nn.Parameter):
            print(f"  A形状: {self.lora_A.shape}, requires_grad={self.lora_A.requires_grad}")
            print(f"  B形状: {self.lora_B.shape}, requires_grad={self.lora_B.requires_grad}")
            print(f"  A范数: {self.lora_A.norm().item():.6f}")
            print(f"  B范数: {self.lora_B.norm().item():.6f}")

    def forward(self, input):
        if self.grad_ckpt:
            return self.ckpt_forward(input)
        return self.no_ckpt_forward(input)

    def ckpt_forward(self, input):
        return torch.utils.checkpoint.checkpoint(self.no_ckpt_forward,
                                                 input,
                                                 use_reentrant=True)

    def no_ckpt_forward(self, input):
        if not self.built_codebook_class:
            self.codebook_class = codebook.get_quantized_class(
                self.codebook_id.item()
            )(self.Qidxs.device)
            if self.codebook_class.codebook.version != self.codebook_version:
                raise Exception(
                    f"Saved weights version ({self.codebook_version}) does not match the "\
                    f"codebook version ({self.codebook_class.codebook.version}). "\
                    "Please download the latest weights from https://huggingface.co/relaxml")

            Qidxs_dev = self.Qidxs.device
            self.Qidxs = self.Qidxs.cpu()
            split_qidxs = self.codebook_class.maybe_unpack_idxs(self.Qidxs)
            self.Qidxs_list = []
            for i in range(len(split_qidxs)):
                self.register_buffer(f'Qidxs_{i}',
                                     split_qidxs[i].to(Qidxs_dev))
                exec(f'self.Qidxs_list.append(self.Qidxs_{i})')
            del self.Qidxs

            # fuse Wscale into SV, legacy code for when Wscale != 1
            # new models have Wscale pre-fused into SV
            self.SV *= self.Wscale

            # cache hadamard transformed manifested weights
            if self.train_mode:
                self.codebook_class.cache_WH(
                    len(self.SU),
                    len(self.SV),
                    self.Qidxs_list,
                    self.had_left_T.T.contiguous(),
                    self.had_right,
                    self.K_left,
                    self.K_right,
                    resid_scale_override=self.resid_scale_override,
                )
                del self.Qidxs_list, self.had_left_T, self.had_right, self.K_left, self.K_right
                self.Qidxs_list = None
                self.had_left_T = None
                self.had_right = None
                self.K_left = None
                self.K_right = None
                clean()

            self.built_codebook_class = True

        # result = self.codebook_class(
        #     input,
        #     self.Qidxs_list,
        #     self.SU,
        #     self.SV,
        #     self.had_left_T,
        #     self.had_right,
        #     self.K_left,
        #     self.K_right,
        #     rank=self.rank,
        #     A=self.A,
        #     B=self.B,
        #     rescale_WH=self.rescale_WH,
        #     scaleWH=self.scaleWH,
        #     packed=self.packed,
        #     resid_scale_override=self.resid_scale_override,
        #     train_mode=self.train_mode).to(input.dtype)

        # 注意：这里不再传递低秩修正的参数
        quant_result = self.codebook_class(
            input,
            self.Qidxs_list,
            self.SU,
            self.SV,
            self.had_left_T,
            self.had_right,
            self.K_left,
            self.K_right,
            rank=self.rank,
            A=None,
            B=None,
            rescale_WH=self.rescale_WH,
            scaleWH=self.scaleWH,
            packed=self.packed,
            resid_scale_override=self.resid_scale_override,
            train_mode=self.train_mode).to(input.dtype)
        # 添加低秩修正
        if self.lora_A is not None and self.lora_B is not None:
            
            # lora前向传播
            lora_hidden = torch.matmul(input, self.lora_A.T)  # (batch_size, rank)
            lora_output = torch.matmul(lora_hidden, self.lora_B.T)  # (batch_size, out_features)
            lora_output = lora_output * self.lora_alpha
            result = quant_result + lora_output 
        else:
            lora_output = 0
            result = quant_result
        if self.has_bias:
            return result + self.bias
        return result


def convert_to_lora_linear(module, rank=16, alpha=16.0, trainable=True, dropout=0.0):
    """
    将现有的QuantizedLinearWithlora转换为新的QuantizedLinearWithLoRA
    
    Args:
        module: 原始的QuantizedLinearWithlora模块
        rank: LoRA秩
        alpha: LoRA缩放因子
        trainable: 是否可训练
        dropout: dropout概率
    
    Returns:
        新的QuantizedLinearWithLoRA模块
    """
    print(f"\n{'='*60}")
    print(f"开始转换模块为 LoRA 版本")
    print(f"  模块类型: {type(module).__name__}")
    print(f"  输入维度: {module.in_features}, 输出维度: {module.out_features}")
    print(f"  LoRA rank: {rank}, alpha: {alpha}")
    print(f"{'='*60}\n")
    
    # 从原始模块的Qidxs形状推断codesz和packsz
    # Qidxs.shape = [out_features, in_features / (codesz * packsz)]
    # 因此: codesz * packsz = in_features / Qidxs.shape[1]
    
    # 尝试从 module.config 获取（如果存在）
    if hasattr(module, 'config'):
        codesz = module.config.get('codesz', 8)
        packsz = module.config.get('packsz', 4)
        idx_dtype = module.config.get('idx_dtype', 'torch.int64')
        print(f"    从 config 获取: codesz={codesz}, packsz={packsz}, idx_dtype={idx_dtype}")
    else:
        # 从 Qidxs 推断
        if hasattr(module, 'Qidxs') and module.Qidxs is not None:
            qidxs_shape = module.Qidxs.shape
            # codesz * packsz = in_features / Qidxs.shape[1]
            code_pack_product = module.in_features // qidxs_shape[1]
            
            # 对于 E8P，通常 codesz=8
            # 对于 E8P12: codesz=8, packsz=4 (总共32)
            # 对于 E8P: codesz=8, packsz=1 (总共8)
            if code_pack_product == 32:
                codesz, packsz = 8, 4
            elif code_pack_product == 8:
                codesz, packsz = 8, 1
            elif code_pack_product == 16:
                codesz, packsz = 8, 2
            else:
                # 默认假设 codesz=8
                codesz = 8
                packsz = code_pack_product // 8
            
            # 从 Qidxs 的 dtype 推断
            idx_dtype = str(module.Qidxs.dtype).replace('torch.', 'torch.')
            if not idx_dtype.startswith('torch.'):
                idx_dtype = 'torch.' + str(module.Qidxs.dtype)
            
            print(f"    从 Qidxs 推断: codesz={codesz}, packsz={packsz}, idx_dtype={idx_dtype}")
            print(f"      (Qidxs.shape={qidxs_shape}, code_pack_product={code_pack_product})")
        else:
            # 使用默认值
            codesz, packsz = 8, 4
            idx_dtype = 'torch.int64'
            print(f"    使用默认值: codesz={codesz}, packsz={packsz}, idx_dtype={idx_dtype}")
    
    # 创建新模块
    new_module = QuantizedLinearWithlora(
        in_features=module.in_features,
        out_features=module.out_features,
        codesz=codesz,
        packsz=packsz,
        pack_out=False,
        idx_dtype=idx_dtype,
        codebook_version=module.codebook_version,
        rank=rank,
        lora_alpha=alpha,
        lora_trainable=trainable,
        rescale_WH=module.rescale_WH,
        bias=module.has_bias,
        resid_scale_override=module.resid_scale_override,
        train_mode=module.train_mode,
        grad_ckpt=module.grad_ckpt,
    )
    
    # 尝试加载 state dict (strict=False 因为新模块有 lora_A/lora_B)
    missing_keys, unexpected_keys = new_module.load_state_dict(module.state_dict(), strict=False)
    if missing_keys:
        print(f"    加载时缺少的键: {missing_keys}")
    if unexpected_keys:
        print(f"    加载时多余的键: {unexpected_keys}")
    
    # 如果原模块有A和B，将它们复制到新的LoRA参数
    if hasattr(module, 'A') and module.A is not None and module.B is not None:
        if rank != module.rank:
            print(f"警告: 原模块rank={module.rank}，新模块rank={rank}，需要重新初始化")
        else:
            if trainable:
                new_module.lora_A.data = module.B.t().contiguous()
                new_module.lora_B.data = module.A.t().contiguous()
            else:
                new_module.lora_A = module.B.t().contiguous()
                new_module.lora_B = module.A.t().contiguous()
    
    return new_module