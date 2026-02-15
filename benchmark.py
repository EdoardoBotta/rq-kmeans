import torch
import triton.testing as testing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from kernels.quantize import quantize_fwd
from kernels.quantize import quantize_fwd_mm

device = "cuda"
torch.manual_seed(0)


def torch_cdist_fn(x, codebook):
    return torch.cdist(x, codebook).argmin(-1).to(torch.int32)

torch_cdist_compiled = torch.compile(torch_cdist_fn, mode="max-autotune")


def run_bench(fn, warmup=25, rep=100):
    return testing.do_bench(fn, warmup=warmup, rep=rep)


def benchmark_grid(N, B_vals, D_vals, dtype=torch.float32):
    records = []

    for B in B_vals:
        for D in D_vals:
            print(f"Running B={B}, D={D}, N={N}")

            x = torch.randn(B, D, device=device, dtype=dtype)
            codebook = torch.randn(N, D, device=device, dtype=dtype)

            torch_cdist_compiled(x, codebook)

            ms_small = run_bench(lambda: quantize_fwd(x, codebook))
            ms_mm = run_bench(lambda: quantize_fwd_mm(x, codebook))
            ms_torch = run_bench(lambda: torch_cdist_compiled(x, codebook))

            records.append({
                "B": B,
                "D": D,
                "NB": N * B,
                "small_speedup": ms_torch / ms_small,
                "mm_speedup": ms_torch / ms_mm,
                "small_latency_ms": ms_small,
                "mm_latency_ms": ms_mm,
                "torch_latency_ms": ms_torch,
            })

    return pd.DataFrame(records)


def plot_heatmap(df, value_col, title, filename):
    pivot = df.pivot(index="NB", columns="D", values=value_col)

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot.sort_index(),
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "Speedup vs torch.compile(cdist)"},
    )
    plt.title(title)
    plt.ylabel("Total Distances (N * B)")
    plt.xlabel("Feature Dim (D)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, format="jpg")
    plt.close()


if __name__ == "__main__":

    N = 512

    B_vals = [1024, 4096, 8192, 16384, 32768]
    D_vals = [16, 32, 64, 128, 256]

    df = benchmark_grid(N=N, B_vals=B_vals, D_vals=D_vals)
    df.to_csv("out/speedup_grid_compiled.csv", index=False)

    plot_heatmap(
        df,
        value_col="small_speedup",
        title=f"quantize_fwd Speedup vs torch.compile(cdist) (N={N})",
        filename="out/heatmap_small_speedup.jpg",
    )

    plot_heatmap(
        df,
        value_col="mm_speedup",
        title=f"quantize_fwd_mm Speedup vs torch.compile(cdist) (N={N})",
        filename="out/heatmap_mm_speedup.jpg",
    )
