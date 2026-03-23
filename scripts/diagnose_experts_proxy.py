#!/usr/bin/env python3
import argparse
import json
import random
import re
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _print_header(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def run_transform_invertibility(device: str):
    _print_header("Check 1: Transform Invertibility")
    result = {"ok": True, "rows": [], "error": None}
    try:
        import torch
        from lib.utils.matmul_had import matmul_hadU, matmul_hadUt

        if device == "auto":
            run_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            run_device = device

        print(f"device = {run_device}")
        # Experts-related representative shapes in this repo:
        # gate/up: (1408, 2048), down: (2048, 1408), shared down often uses 2816.
        for m, n in [
            (64, 2048),
            (64, 1408),
            (64, 2816),
            (1408, 2048),
            (2048, 1408),
            (2048, 2816),
        ]:
            try:
                torch.manual_seed(0)
                W = torch.randn(m, n, device=run_device, dtype=torch.float32)
                SU = torch.where(
                    torch.randn(n, device=run_device) > 0,
                    torch.ones(n, device=run_device),
                    -torch.ones(n, device=run_device),
                )
                SV = torch.where(
                    torch.randn(m, device=run_device) > 0,
                    torch.ones(m, device=run_device),
                    -torch.ones(m, device=run_device),
                )

                # Base sanity: U(Ut(x)) should approximately recover x.
                X = torch.randn(32, n, device=run_device, dtype=torch.float32)
                Xrec = matmul_hadU(matmul_hadUt(X))
                base_rel = ((Xrec - X).norm() / X.norm()).item()

                # Forward transform must match lib/algo/quip.py::RHT_W exactly.
                Wr = matmul_hadUt((matmul_hadUt(W.T * SV).T) * SU)
                Wrec = (matmul_hadU((matmul_hadU(Wr) * SU).T) * SV).T

                rel = ((Wrec - W).norm() / W.norm()).item()
                mx = (Wrec - W).abs().max().item()
                row = {"m": m, "n": n, "rel": rel, "max": mx, "base_rel": base_rel}
                result["rows"].append(row)
                print(f"m={m}, n={n}, base_rel={base_rel:.6e}, rel={rel:.6e}, max={mx:.6e}")
            except Exception as ex:  # noqa: BLE001
                result["ok"] = False
                row = {"m": m, "n": n, "error": repr(ex)}
                result["rows"].append(row)
                print(f"FAILED m={m}, n={n}, error={repr(ex)}")
                traceback.print_exc()
    except Exception as ex:  # noqa: BLE001
        result["ok"] = False
        result["error"] = repr(ex)
        print(f"FAILED import/runtime: {repr(ex)}")
        traceback.print_exc()
    return result


def run_hessian_dim_scan(experts_hessian_dir: Path, sample_size: int):
    _print_header("Check 2: Hessian Dimension Distribution")
    result = {
        "ok": True,
        "exists": experts_hessian_dir.exists(),
        "total_files": 0,
        "sample_count": 0,
        "stats": {},
        "error": None,
    }
    if not experts_hessian_dir.exists():
        result["ok"] = False
        print(f"Directory not found: {experts_hessian_dir}")
        return result

    try:
        import torch

        files = list(experts_hessian_dir.glob("*.pt"))
        result["total_files"] = len(files)
        print(f"total_files = {len(files)}")
        if not files:
            result["ok"] = False
            return result

        random.seed(0)
        sample = random.sample(files, min(sample_size, len(files)))
        result["sample_count"] = len(sample)

        stats = {
            "dense_down_proj": [],
            "dense_gate_proj": [],
            "dense_up_proj": [],
            "shared_down_proj": [],
            "shared_gate_proj": [],
            "shared_up_proj": [],
            "expert_down_proj": [],
            "expert_gate_proj": [],
            "expert_up_proj": [],
        }
        ct_stats = {
            "dense_down_proj": [],
            "dense_gate_proj": [],
            "dense_up_proj": [],
            "shared_down_proj": [],
            "shared_gate_proj": [],
            "shared_up_proj": [],
            "expert_down_proj": [],
            "expert_gate_proj": [],
            "expert_up_proj": [],
        }

        for p in sample:
            pack = torch.load(p, map_location="cpu")
            n = int(pack["n"])
            ct = int(pack.get("ct", -1))
            name = p.name
            if "expert" in name and "down_proj" in name:
                stats["expert_down_proj"].append(n)
                if ct >= 0:
                    ct_stats["expert_down_proj"].append(ct)
            elif "expert" in name and "gate_proj" in name:
                stats["expert_gate_proj"].append(n)
                if ct >= 0:
                    ct_stats["expert_gate_proj"].append(ct)
            elif "expert" in name and "up_proj" in name:
                stats["expert_up_proj"].append(n)
                if ct >= 0:
                    ct_stats["expert_up_proj"].append(ct)
            elif "shared" in name and "down_proj" in name:
                stats["shared_down_proj"].append(n)
                if ct >= 0:
                    ct_stats["shared_down_proj"].append(ct)
            elif "shared" in name and "gate_proj" in name:
                stats["shared_gate_proj"].append(n)
                if ct >= 0:
                    ct_stats["shared_gate_proj"].append(ct)
            elif "shared" in name and "up_proj" in name:
                stats["shared_up_proj"].append(n)
                if ct >= 0:
                    ct_stats["shared_up_proj"].append(ct)
            elif "dense" in name and "down_proj" in name:
                stats["dense_down_proj"].append(n)
                if ct >= 0:
                    ct_stats["dense_down_proj"].append(ct)
            elif "dense" in name and "gate_proj" in name:
                stats["dense_gate_proj"].append(n)
                if ct >= 0:
                    ct_stats["dense_gate_proj"].append(ct)
            elif "dense" in name and "up_proj" in name:
                stats["dense_up_proj"].append(n)
                if ct >= 0:
                    ct_stats["dense_up_proj"].append(ct)

        short_stats = {}
        for k, values in stats.items():
            cts = ct_stats[k]
            ct_summary = None
            if cts:
                ct_summary = {
                    "min": int(min(cts)),
                    "median": int(sorted(cts)[len(cts) // 2]),
                    "max": int(max(cts)),
                }
            short_stats[k] = {
                "count": len(values),
                "unique_n": sorted(set(values)),
                "ct": ct_summary,
            }
            print(
                f"{k}: count={len(values)}, unique_n={sorted(set(values))}, "
                f"ct={ct_summary}"
            )

        result["stats"] = short_stats
    except Exception as ex:  # noqa: BLE001
        result["ok"] = False
        result["error"] = repr(ex)
        print(f"FAILED: {repr(ex)}")
        traceback.print_exc()

    return result


def run_quant_cache_scan(quant_dir: Path):
    _print_header("Check 3: Quant Cache SU/SV Shape & Finite")
    result = {
        "ok": True,
        "exists": quant_dir.exists(),
        "scan_files": 0,
        "bad_count": 0,
        "bad_examples": [],
        "error": None,
    }
    if not quant_dir.exists():
        result["ok"] = False
        print(f"Directory not found: {quant_dir}")
        return result

    try:
        import torch

        patterns = [
            "*_expert*_*.pt",
            "*_shared_*_proj.pt",
            "*_dense_*_proj.pt",
        ]
        files = []
        for pat in patterns:
            files.extend(sorted(quant_dir.glob(pat)))

        result["scan_files"] = len(files)
        print(f"scan_files = {len(files)}")

        bad = []
        for p in files:
            d = torch.load(p, map_location="cpu")
            if "shapes" not in d or "SU" not in d or "SV" not in d:
                continue
            m, n = d["shapes"][0]
            su = d["SU"]
            sv = d["SV"]

            ok_shape = (su.numel() == n and sv.numel() == m)
            ok_finite = bool(torch.isfinite(su).all().item() and torch.isfinite(sv).all().item())
            if (not ok_shape) or (not ok_finite):
                bad.append(
                    {
                        "file": p.name,
                        "su_shape": tuple(su.shape),
                        "sv_shape": tuple(sv.shape),
                        "weight_shape": (m, n),
                        "finite": ok_finite,
                    }
                )

        result["bad_count"] = len(bad)
        result["bad_examples"] = bad[:80]
        print(f"bad_count = {len(bad)}")
        for row in bad[:80]:
            print(row)
    except Exception as ex:  # noqa: BLE001
        result["ok"] = False
        result["error"] = repr(ex)
        print(f"FAILED: {repr(ex)}")
        traceback.print_exc()

    return result


def run_proxy_log_parse(proxy_log: Path):
    _print_header("Check 4: Parse Experts Proxy Error")
    result = {
        "ok": True,
        "exists": proxy_log.exists(),
        "count": 0,
        "top": [],
        "error": None,
    }
    if not proxy_log.exists():
        result["ok"] = False
        print(f"Log not found: {proxy_log}")
        return result

    try:
        pat = re.compile(r"(.+?) proxy error:\s*([0-9.eE+-]+)")
        rows = []
        with proxy_log.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if not m:
                    continue
                msg, val = m.group(1), float(m.group(2))
                if ("expert" in msg) or ("shared" in msg) or ("dense_" in msg):
                    rows.append((val, msg.strip()))

        rows.sort(reverse=True, key=lambda x: x[0])
        result["count"] = len(rows)
        result["top"] = [{"proxy": v, "msg": msg} for v, msg in rows[:80]]

        print(f"experts_related_proxy_count = {len(rows)}")
        for v, msg in rows[:80]:
            print(f"{v:.6f} | {msg}")
    except Exception as ex:  # noqa: BLE001
        result["ok"] = False
        result["error"] = repr(ex)
        print(f"FAILED: {repr(ex)}")
        traceback.print_exc()

    return result


def main():
    parser = argparse.ArgumentParser(description="Diagnose high proxy error on DeepSeek experts quantization")
    parser.add_argument(
        "--experts-hessian-dir",
        type=Path,
        default=Path("/fact_home/zeyuli/quip_sharp/hessians_deepseek_moe_16b_base_experts"),
        help="Path to experts hessian folder",
    )
    parser.add_argument(
        "--quant-dir",
        type=Path,
        default=None,
        help="Quantized output folder (save_path). If omitted, Check 3 is skipped.",
    )
    parser.add_argument(
        "--proxy-log",
        type=Path,
        default=None,
        help="Quantization log file containing 'proxy error:' lines. If omitted, Check 4 is skipped.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Number of Hessian files to sample for dimension scan",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for invertibility test",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output json path for machine-readable result",
    )
    args = parser.parse_args()

    summary = {}
    summary["check1_transform"] = run_transform_invertibility(args.device)
    summary["check2_hessian_dims"] = run_hessian_dim_scan(args.experts_hessian_dir, args.sample_size)

    if args.quant_dir is not None:
        summary["check3_quant_cache"] = run_quant_cache_scan(args.quant_dir)
    else:
        print("\nSkip Check 3: --quant-dir not provided")

    if args.proxy_log is not None:
        summary["check4_proxy_log"] = run_proxy_log_parse(args.proxy_log)
    else:
        print("Skip Check 4: --proxy-log not provided")

    _print_header("Summary")
    print(json.dumps(summary, indent=2, ensure_ascii=True))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"saved json: {args.json_out}")


if __name__ == "__main__":
    main()
