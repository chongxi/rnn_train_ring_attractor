#include "cuda_common.cuh"

struct Config {
    int N = 512;
    int a_dim = 32;
    int seq_len = 4;
    int batch_size = 64;
    int activation_type = 3;
    float J0 = 1.0f;
    float J1 = 0.5f;
    float alpha = 0.15f;
    bool run_check = true;
    bool run_benchmark = false;
    int timing_iterations = 100;
};

void print_usage(const char* prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  --N <int>              Number of neurons (default: 128)\n");
    printf("  --a_dim <int>          Action dimension (default: 128)\n");
    printf("  --seq_len <int>        Sequence length (default: 2)\n");
    printf("  --batch_size <int>     Batch size (default: 64)\n");
    printf("  --activation <string>  Activation type: relu, gelu, tanh, silu (default: silu)\n");
    printf("  --J0 <float>           J0 parameter (default: 1.0)\n");
    printf("  --J1 <float>           J1 parameter (default: 0.5)\n");
    printf("  --alpha <float>        Alpha parameter (default: 0.15)\n");
    printf("  --check <0|1>          Run correctness check (default: 1)\n");
    printf("  --benchmark <0|1>      Run benchmark (default: 0)\n");
    printf("  --iters <int>          Benchmark iterations (default: 100)\n");
    printf("  --help                 Show this help message\n");
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) { print_usage(argv[0]); exit(0); }
        else if (strcmp(argv[i], "--N") == 0 && i + 1 < argc) { cfg.N = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--a_dim") == 0 && i + 1 < argc) { cfg.a_dim = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--seq_len") == 0 && i + 1 < argc) { cfg.seq_len = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--batch_size") == 0 && i + 1 < argc) { cfg.batch_size = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--activation") == 0 && i + 1 < argc) {
            std::string act = argv[++i];
            if (act == "relu") cfg.activation_type = 0;
            else if (act == "gelu") cfg.activation_type = 1;
            else if (act == "tanh") cfg.activation_type = 2;
            else if (act == "silu") cfg.activation_type = 3;
            else { printf("Unknown activation: %s\n", act.c_str()); exit(1); }
        }
        else if (strcmp(argv[i], "--J0") == 0 && i + 1 < argc) { cfg.J0 = atof(argv[++i]); }
        else if (strcmp(argv[i], "--J1") == 0 && i + 1 < argc) { cfg.J1 = atof(argv[++i]); }
        else if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) { cfg.alpha = atof(argv[++i]); }
        else if (strcmp(argv[i], "--check") == 0 && i + 1 < argc) { cfg.run_check = atoi(argv[++i]) != 0; }
        else if (strcmp(argv[i], "--benchmark") == 0 && i + 1 < argc) { cfg.run_benchmark = atoi(argv[++i]) != 0; }
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) { cfg.timing_iterations = atoi(argv[++i]); }
        else { printf("Unknown argument: %s\n", argv[i]); print_usage(argv[0]); exit(1); }
    }
    return cfg;
}

void print_config(const Config& cfg) {
    const char* act_names[] = {"relu", "gelu", "tanh", "silu"};
    printf("Configuration:\n");
    printf("  N: %d, a_dim: %d, seq_len: %d, batch_size: %d\n", cfg.N, cfg.a_dim, cfg.seq_len, cfg.batch_size);
    printf("  activation: %s, J0: %.2f, J1: %.2f, alpha: %.2f\n", act_names[cfg.activation_type], cfg.J0, cfg.J1, cfg.alpha);
    printf("  check: %s, benchmark: %s", cfg.run_check ? "true" : "false", cfg.run_benchmark ? "true" : "false");
    if (cfg.run_benchmark) printf(", iterations: %d", cfg.timing_iterations);
    printf("\n\n");
}

bool allClose(const std::vector<float>& mat_ref, const std::vector<float>& mat_impl,
              float rtol = 1e-05f, float atol = 1e-08f, int max_print = 5) {
    if (mat_ref.size() != mat_impl.size()) {
        std::cout << "Size mismatch: " << mat_ref.size() << " vs " << mat_impl.size() << std::endl;
        return false;
    }

    int num_mismatched = 0;
    int printed_count = 0;

    for (size_t i = 0; i < mat_ref.size(); i++) {
        float a = mat_ref[i];
        float b = mat_impl[i];

        // Handle NaN and infinity cases
        if (std::isnan(a) || std::isnan(b)) {
            if (!(std::isnan(a) && std::isnan(b))) {
                num_mismatched++;
                if (printed_count < max_print) {
                    std::cout << "NaN at index " << i << ": ref=" << a << " vs impl=" << b << std::endl;
                    printed_count++;
                }
            }
            continue;
        }

        if (std::isinf(a) || std::isinf(b)) {
            if (a != b) {
                num_mismatched++;
                if (printed_count < max_print) {
                    std::cout << "Infinity at index " << i << ": ref=" << a << " vs impl=" << b << std::endl;
                    printed_count++;
                }
            }
            continue;
        }

        // PyTorch allclose formula: |a - b| <= atol + rtol * |b|
        float diff = std::abs(a - b);
        float tol = atol + rtol * std::abs(b);

        if (diff > tol) {
            num_mismatched++;
            if (printed_count < max_print) {
                std::cout << "Mismatch at idx " << i << ": "
                         << "ref=" << a << " vs impl=" << b << " (diff: " << diff
                         << ", tol: " << tol << ")" << std::endl;
                printed_count++;
            }
        }
    }

    if (num_mismatched > 0) {
        std::cout << "...\n";
        std::cout << "Total mismatched elements: " << num_mismatched << " out of " << mat_ref.size() << std::endl;
        return false;
    }

    return true;
}

template<typename KernelFunc, typename... Args>
bool check_correctness(const char* kernel_name, KernelFunc kernel,
                      std::vector<float>& matC_cpu,std::vector<float>& matC_gpu, float* d_C_out,
                      dim3 gridSize, dim3 blockSize, Args... args) {
    // d_C_out is part of "args...", I needed to expose it to copy back the result
    int M_N = matC_cpu.size();

    kernel<<<gridSize, blockSize>>>(args...);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(matC_gpu.data(), d_C_out, M_N * sizeof(float), cudaMemcpyDeviceToHost));

    // float atol = 1e-3;
    // float rtol = 1e-5;

    float atol = 0.1f;
    float rtol = 0.01f;

    int max_print = 10;
    // PyTorch allclose formula: |a - b| <= atol + rtol * |b|
    bool correct = allClose(matC_cpu, matC_gpu, rtol, atol, max_print);
    printf("%s: %s\n", kernel_name, correct ? "PASSED" : "FAILED");
    return correct;
}


template<typename LauncherFunc, typename... Args>
void benchmark_launcher(const char* launcher_name, LauncherFunc launcher, int num_runs, Args... args) {
    float* times = new float[num_runs];

    // Warmup
    launcher(args...);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    if (num_runs == 0) return;

    for(int i = 0; i < num_runs; i++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        launcher(args...);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&times[i], start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    float sum = 0, min_time = times[0], max_time = times[0];
    for(int i = 0; i < num_runs; i++) {
        sum += times[i];
        if(times[i] < min_time) min_time = times[i];
        if(times[i] > max_time) max_time = times[i];
    }
    float avg = sum / num_runs;

    float variance = 0;
    for(int i = 0; i < num_runs; i++) {
        variance += (times[i] - avg) * (times[i] - avg);
    }
    float std_dev = sqrt(variance / num_runs);

    printf("%s:\n", launcher_name);
    printf("%.3f ± %.3f ms\n", avg, std_dev);
    printf("min: %.3f ms, max: %.3f ms\n\n", min_time, max_time);

    delete[] times;
}

