"""Conservative execution limits, applied before numerical imports."""

import os


THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
SAFE_MAX_RECORDS = 300
CUDA_MIN_FREE_MIB = {"inference": 4096, "training": 5000}
CUDA_MIN_TOTAL_MIB = 6000
CUDA_MEMORY_FRACTION = 0.70
_TORCH_CONFIGURED = False


def prepare_environment(threads=1, device="cpu"):
    if not 1 <= threads <= 2:
        raise ValueError("Use one or two CPU threads; unrestricted parallelism is disabled")
    for name in THREAD_VARIABLES:
        os.environ[name] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
    if device == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
    if hasattr(os, "nice"):
        os.nice(5)


def validate_request(args):
    if not 1 <= args.threads <= 2:
        raise ValueError("CPU threads must be 1 or 2")
    if not 1 <= getattr(args, "batch_size", 2) <= 4:
        raise ValueError("Batch size must be between 1 and 4")
    if getattr(args, "epochs", 1) < 1:
        raise ValueError("Epochs must be positive")
    device = getattr(args, "device", "cpu")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda")
    intensive = getattr(args, "allow_intensive", False)
    if device == "cuda" and not intensive:
        raise ValueError("CUDA requires --allow-intensive and a VRAM preflight")
    if device == "cuda" and getattr(args, "batch_size", 2) != 1:
        raise ValueError("Guarded CUDA mode requires --batch-size 1")
    if not intensive:
        if args.threads != 1 or getattr(args, "batch_size", 2) > 2:
            raise ValueError("Safe mode requires one CPU thread and batch size at most 2")
        if getattr(args, "fine_tune", False) or args.command == "finetune":
            raise ValueError("Fine-tuning requires --allow-intensive; it is disabled in safe mode")
        if getattr(args, "profile", "smoke") != "smoke":
            raise ValueError("Pilot/full profiles require --allow-intensive; use the smoke profile")
        if getattr(args, "records", 100) * 3 > SAFE_MAX_RECORDS:
            raise ValueError("Safe mode permits at most 100 records per difficulty")


def check_record_budget(count, allow_intensive=False):
    if not allow_intensive and count > SAFE_MAX_RECORDS:
        raise ValueError(f"Safe mode accepts at most {SAFE_MAX_RECORDS} PRs; use a new small run instead of the interrupted pilot")


def validate_vram_capacity(free_mib, total_mib, purpose):
    if purpose not in CUDA_MIN_FREE_MIB:
        raise ValueError(f"Unknown CUDA workload: {purpose}")
    required_free = CUDA_MIN_FREE_MIB[purpose]
    if total_mib < CUDA_MIN_TOTAL_MIB:
        raise RuntimeError(f"CUDA {purpose} requires at least {CUDA_MIN_TOTAL_MIB} MiB total VRAM; found {total_mib} MiB")
    if free_mib < required_free:
        raise RuntimeError(f"CUDA {purpose} needs at least {required_free} MiB free VRAM; found {free_mib} MiB. No model was loaded.")
    return {"purpose": purpose, "free_mib": int(free_mib), "total_mib": int(total_mib),
            "minimum_free_mib": required_free, "process_memory_fraction": CUDA_MEMORY_FRACTION}


def preflight_cuda(purpose):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but a working CUDA device is unavailable. No model was loaded.")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    report = validate_vram_capacity(free_bytes // (1024 * 1024), total_bytes // (1024 * 1024), purpose)
    torch.cuda.set_per_process_memory_fraction(CUDA_MEMORY_FRACTION, torch.cuda.current_device())
    report["gpu_name"] = torch.cuda.get_device_name()
    return report


def configure_torch():
    global _TORCH_CONFIGURED
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    if not _TORCH_CONFIGURED:
        torch.set_num_interop_threads(1)
        _TORCH_CONFIGURED = True