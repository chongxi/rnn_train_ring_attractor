import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


torch.set_printoptions(linewidth=200)
np.set_printoptions(
    precision=4,      # 6 decimal places
    suppress=True,    # Don't use scientific notation for small numbers
    linewidth=200,    # Wider lines
    formatter={'float': lambda x: f'{x:>12.4f}'})

def check_tensor_match(tsr_impl, tsr_ref, name, rtol=1e-5, atol=1e-8, max_print=1):
    if not torch.allclose(tsr_impl, tsr_ref, rtol=rtol, atol=atol):
        print(f"\n{name} differences: a_tol = {atol}, r_tol = {rtol}")
        diff = (tsr_impl - tsr_ref).abs()
        rdiff = diff / (torch.abs(tsr_ref) + atol)
        mismatch = ~torch.isclose(tsr_impl, tsr_ref, rtol=rtol, atol=atol)
        num_mismatched = mismatch.sum().item()
        total_elements = tsr_impl.numel()
        percentage = (num_mismatched / total_elements) * 100
        all_indices = torch.nonzero(mismatch)
        first_indices = all_indices[:max_print]
        last_indices = all_indices[-max_print:] if len(all_indices) > max_print else torch.empty(0, all_indices.shape[1], dtype=torch.long)

        print("Mismatch at                 ref         impl        diff        rdiff")
        print("---------------------------------------------------------------------")
        print("First mismatches:")
        for idx in first_indices:
            idx_str = ','.join([f"{i}" for i in idx.tolist()])
            idx_tuple = tuple(idx.tolist())
            print(f"[{idx_str}]: {tsr_ref[idx_tuple]:10.6f} {tsr_impl[idx_tuple]:10.6f} {diff[idx_tuple]:10.6f} {rdiff[idx_tuple]:10.6f}")

        if len(all_indices) > max_print:
            print("...")
            print("Last mismatches:")
            for idx in last_indices:
                idx_str = ','.join([f"{i}" for i in idx.tolist()])
                idx_tuple = tuple(idx.tolist())
                print(f"[{idx_str}]: {tsr_ref[idx_tuple]:10.6f} {tsr_impl[idx_tuple]:10.6f} {diff[idx_tuple]:10.6f} {rdiff[idx_tuple]:10.6f}")

        print(f"Total mismatched elements: {num_mismatched} out of {total_elements} ({percentage:.1f}%)")
        print(f"diff: {diff.mean():.6f} ± {diff.std():.6f}")
        print(f"rdiff: {rdiff.mean():.6f} ± {rdiff.std():.6f}")
        return False
    else:
        print(f"{name} match!")
        return True

def measure_latency_cuda(fn, *args, n_warmup=2, n_iters=20, **kwargs):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(n_warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        start_event.record()
        fn(*args, **kwargs)
        end_event.record()

        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))  # ms

    times = torch.tensor(times, device="cpu")
    mean = times.mean().item()
    std = times.std(unbiased=False).item()
    return f"{mean:.3f} ± {std:.3f} ms"