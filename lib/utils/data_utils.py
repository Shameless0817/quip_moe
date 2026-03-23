import multiprocessing as mp

import glog
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from lib import codebook

from .matmul_had import matmul_hadU


def flat_to_sym(V, N):
    A = torch.zeros(N, N, dtype=V.dtype, device=V.device)
    idxs = torch.tril_indices(N, N, device=V.device)
    A[idxs.unbind()] = V
    A[idxs[1, :], idxs[0, :]] = V
    return A


def sym_to_flat(A):
    N = A.shape[-1]
    idxs = torch.tril_indices(N, N, device=A.device)
    return A[idxs.unbind()]


def register_H_hook(module, device):
    # Get input dimension - handle different module types
    if hasattr(module, 'in_features'):
        # Standard nn.Linear module
        n = module.in_features
    elif hasattr(module, 'weight'):
        # For modules like MoEGate that have weight but not in_features
        # weight shape is typically [out_features, in_features]
        n = module.weight.shape[1]
    elif hasattr(module, 'gating_dim'):
        # Fallback for modules with gating_dim attribute
        n = module.gating_dim
    else:
        raise ValueError(f"Cannot determine input dimension for module type {type(module)}")
    
    H = torch.zeros(n, n, dtype=torch.float64, device=device)
    mu = torch.zeros(n, dtype=torch.float64, device=device)
    ct = 0

    def H_hook(module, x):
        nonlocal H, mu, ct, n
        x = x[0].reshape(-1, n).to(torch.float64)
        mu.add_(x.sum(dim=0))
        H.addmm_(x.T, x)
        ct += len(x)

    hook = module.register_forward_pre_hook(H_hook)

    def done():
        nonlocal H, mu, ct, hook
        hook.remove()
        return H.cpu(), mu.cpu(), ct

    return done


def wrap_tokenizer(tokenizer, x, ctx_size):
    return tokenizer(x,
                     return_tensors='pt',
                     truncation=True,
                     padding=True,
                     max_length=ctx_size)


def sample_rp1t(tokenizer, size=128, ctx_size=2048, nproc=1):
    dataset = load_dataset('ZengXiangyu/RedPajama-Data-1T-Sample',
                           split='train')
    devset = torch.zeros((size, ctx_size), dtype=torch.int64)
    saved = 0
    
    # Note: Using multiprocessing with tokenizer is problematic due to serialization issues.
    # Force single-process mode for reliability.
    while saved < size:
        # Sample random indices
        sample_indices = torch.randint(len(dataset), (min(size - saved, 32),))
        batch_texts = [dataset[int(idx)]['text'] for idx in sample_indices]
        
        try:
            tokens = tokenizer(batch_texts,
                               return_tensors='pt',
                               truncation=True,
                               padding=True,
                               max_length=ctx_size)
            lens = tokens.attention_mask.sum(dim=-1)
            good = torch.where(lens == ctx_size)[0]
            if len(good) > 0:
                if saved + len(good) > size:
                    good = good[:size - saved]
                devset[saved:saved + len(good)] = tokens.input_ids[good]
                saved += len(good)
        except Exception as e:
            continue
    
    return devset


def sample_falcon_refinedweb(tokenizer, size=128, ctx_size=2048, nproc=1):
    dataset = load_dataset('tiiuae/falcon-refinedweb',
                           streaming=True,
                           split='train')
    dataset = dataset.shuffle(buffer_size=100000, seed=0)
    iter_dataset = iter(dataset)

    devset = torch.zeros((size, ctx_size), dtype=torch.int64)
    saved = 0

    p = mp.Pool(nproc)
    while saved < size:
        seqs = [(tokenizer,
                 [next(iter_dataset)['content']
                  for _ in range(size)], ctx_size) for _ in range(nproc)]
        tokens = p.starmap(wrap_tokenizer, seqs)
        for token in tokens:
            good = torch.where(token.attention_mask.sum(dim=-1) == ctx_size)[0]
            if saved + len(good) > size:
                good = good[:size - saved]
            devset[saved:saved + len(good)] = token.input_ids[good]
            saved += len(good)
    p.close()
    return devset


def load_quip(save_name, cb, args, device, return_scales=False):
    glog.info(f"loading cached compressed layer from path \"{save_name}\"")
    dict_loaded = torch.load(save_name,
                             map_location=torch.device('cuda', device))
    SU = dict_loaded['SU'].to(device)
    SV = dict_loaded['SV'].to(device)
    Wscale = dict_loaded['Wscale'].to(device)
    Qidxs = dict_loaded['Qidxs'].to(device)
    n, m = len(SU), len(SV)
    hatWr = cb.to(device).by_idxs(Qidxs, packed=(cb.packsz != 1)).view(m, n)
    hatWr = hatWr * Wscale
    del Wscale
    if args.lora_rank > 0:
        A = dict_loaded['A'].to(device)
        B = dict_loaded['B'].to(device)
        hatWr = hatWr + A @ B
        del A, B
    hatW = (matmul_hadU((matmul_hadU(hatWr) * SU).T) * SV).T
    del SU, SV
    if args.rescale_WH:
        hatW = hatW / dict_loaded['scaleWH'][None, :].to(device)

    if return_scales:
        scale_dict = {}
        for key in dict_loaded:
            if key.endswith('scale'):
                scale_dict[key] = dict_loaded[key]
        return hatW, scale_dict

    return hatW


def unpack_quip(
    module, 
    saved_layer, 
    codebook_id, 
    codesz, 
    codebook_version=None
):
    (m, n) = saved_layer['Qidxs'].shape
    if codebook_id in codebook.cache_permute_set:
        module.Qidxs.copy_(saved_layer['Qidxs'].view(
            m, n // codesz, codesz).permute(1, 0, 2).reshape(m,
                                                             n).contiguous())
    else:
        module.Qidxs.copy_(saved_layer['Qidxs'])
    if module.rank > 0:
        module.A.copy_(saved_layer['A'])
        module.B.copy_(saved_layer['B'])
    module.SU.copy_(saved_layer['SU'])
    module.SV.copy_(saved_layer['SV'])
    if module.rescale_WH:
        module.scaleWH.copy_(saved_layer['scaleWH'])

    module.codebook_id.copy_(codebook_id)
    # 从保存的权重中获取codebook_version
    if codebook_version is not None:
        module.codebook_version = codebook_version
    elif 'codebook_version' in saved_layer:
        module.codebook_version = saved_layer['codebook_version']


def dtype_from_str(str):
    dtype_map = {
        'torch.int64': torch.int64,
        'torch.int32': torch.int32,
        'torch.int16': torch.int16,
        'torch.uint8': torch.uint8,
        # 支持简短格式
        'int64': torch.int64,
        'int32': torch.int32,
        'int16': torch.int16,
        'uint8': torch.uint8,
    }
    return dtype_map[str]


class SimpleDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def split_data(X, Y, args):
    split = int(len(X) - args.ft_valid_size)
    glog.info(f'using {split} training seqs, {len(X) - split} validation seqs')
    train_ds = SimpleDataset(X[:split], Y[:split])
    valid_ds = SimpleDataset(X[split:], Y[split:])
    train_dl = DataLoader(train_ds,
                          batch_size=args.ft_bs,
                          pin_memory=True,
                          shuffle=True)
    valid_dl = DataLoader(valid_ds,
                          batch_size=args.ft_bs,
                          pin_memory=True,
                          shuffle=False)
    return train_dl, valid_dl


def calculate_logits(model, devset, batch_size):
    logits = []
    for i in range(len(devset) // batch_size):
        logits.append(
            model(devset[i * batch_size:(i + 1) *
                         batch_size].cuda())['logits'].cpu())
    logits = torch.concat(logits, dim=0)
    return logits
