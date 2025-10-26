# Ring RNN Performance Benchmark Results (RTX PRO 6000 Blackwell)

| B_size | N | a_dim | Seq len | **torch.sum (ms)** | **torch.matmul (ms)** | **CUDA (ms)** | **CUDA WMMA (ms)** | **vs .sum** | **vs .matmul** |
|:-------:|:--:|:------:|:---------:|:------------------:|:----------------------:|:---------------:|:------------------:|:-------------:|:---------------:|
| 256 | 128  | 16 | 20 | 9.438 ± 0.008 | 2.234 ± 0.015 | 0.372 ± 0.011 | **0.282 ± 0.004** | **33.5×** | **7.9×** |
| 256 | 256  | 16 | 20 | 32.742 ± 0.018 | 2.846 ± 0.007 | 0.897 ± 0.007 | **0.595 ± 0.004** | **55.1×** | **4.8×** |
| 256 | 384  | 16 | 20 | 73.732 ± 0.050 | 8.607 ± 0.021 | 1.755 ± 0.030 | **1.269 ± 0.015** | **58.1×** | **6.8×** |
| 256 | 512  | 16 | 20 | 130.132 ± 0.080 | 16.066 ± 0.009 | 2.795 ± 0.068 | **1.772 ± 0.026** | **73.4×** | **9.1×** |
| 256 | 1024 | 16 | 20 | 755.812 ± 0.141 | 61.446 ± 0.023 | 10.172 ± 0.091 | **6.163 ± 0.198** | **122.6×** | **10.0×** |
| 256 | 2048 | 16 | 20 | 3027.348 ± 0.138 | 241.135 ± 0.094 | 46.110 ± 0.395 | **24.455 ± 0.413** | **123.8×** | **9.9×** |
| 256 | 2048 | 32 | 20 | OOM | 244.381 ± 0.146 | 76.646 ± 0.392 | **36.741 ± 0.496** | — | **6.6×** |
| 256 | 2048 | 64 | 20 | OOM | 251.123 ± 0.014 | 139.247 ± 0.406 | **62.792 ± 0.392** | — | **4.0×** |
| 256 | 1024 | 2  | 20 | 116.973 ± 0.139 | 59.912 ± 0.179 | **10.112 ± 0.153** | N/A | **11.6×** | **5.9×** |
| 256 | 1024 | 3  | 20 | 145.126 ± 0.044 | 59.974 ± 0.033 | **10.129 ± 0.090** | N/A | **14.3×** | **5.9×** |
| 256 | 1024 | 6  | 20 | 231.524 ± 0.056 | 60.584 ± 0.077 | **10.744 ± 0.092** | N/A | **21.6×** | **5.6×** |
| 256 | 1024 | 12 | 20 | 553.342 ± 0.202 | 61.159 ± 0.015 | **10.549 ± 0.094** | N/A | **52.5×** | **5.8×** |

where:
- torch.sum is the original code
- torch.matmul is replacing `Wa_weighted = torch.sum(A_t_expanded * self.Wa.unsqueeze(0), dim=1)` in original code with `torch.matmul(A_t, Wa_flat).view(batch_size, N, N)`
- CUDA is the kernel that generalized to all shapes, it doesn't use tensor cores.
- CUDA WMMA is the kernel using tensor cores that only activates when shapes are divisible by 16

Takeaways:
- CUDA WMMA kernel have geometric mean speedup of **69.2x** over original code
- CUDA WMMA kernel with a_dim = 16 has latency of 6.163 ms is faster than CUDA kernel with a_dim = 2,3,6,12 with latency of 10.1 ms. Therefore, it's better to padd a_dim with zeros to make it divisible by 16 for best speedup.
- CUDA kernel has slightly better accuracy than CUDA WMMA kernel.