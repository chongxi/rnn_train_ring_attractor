#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

int main() {
    int devId;
    cudaGetDevice(&devId);
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devId);
    
    cout << "=== CUDA Device Information ===" << endl;
    cout << "Device ID: " << devId << endl;
    cout << "Device Name: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << endl;
    
    // Critical parameters for CUDA kernel engineers
    int value;
    

    
    // Memory information
    int smemSize, numProcs;
    cudaDeviceGetAttribute(&smemSize, cudaDevAttrMaxSharedMemoryPerBlock, devId);
    cudaDeviceGetAttribute(&numProcs, cudaDevAttrMultiProcessorCount, devId);
    
    cout << endl << "=== Memory Information ===" << endl;
    cout << "Shared Memory per Block: " << smemSize << " bytes (" << smemSize/1024 << " KB)" << endl;
    
    cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerMultiprocessor, devId);
    cout << "Shared Memory per SM: " << value << " bytes (" << value/1024 << " KB)" << endl;
    
    cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << endl;
    cout << "Constant Memory: " << prop.totalConstMem << " bytes" << endl;
    
    cudaDeviceGetAttribute(&value, cudaDevAttrL2CacheSize, devId);
    cout << "L2 Cache Size: " << value << " bytes (" << value/1024 << " KB)" << endl;
    
    cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << endl;
    cout << "Memory Clock Rate: " << prop.memoryClockRate << " KHz" << endl;
    
    // Calculate memory bandwidth (theoretical peak)
    // Bandwidth = (Memory Clock Rate * 2) * (Bus Width / 8) / 1000000 GB/s
    // Factor of 2 for DDR (Double Data Rate)
    double memoryBandwidth = (static_cast<double>(prop.memoryClockRate) * 2.0 * prop.memoryBusWidth) / (8.0 * 1000000.0);
    cout << "Memory Bandwidth: " << memoryBandwidth << " GB/s" << endl;
    
    // Compute and processing info
    cout << endl << "=== Compute Information ===" << endl;

    // Thread and Block limits
    cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, devId);
    cout << "Max Threads per Block: " << value << endl;
    
    cout << "Max Block Dimensions: " << prop.maxThreadsDim[0] << " x " 
         << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << endl;
    
    cout << "Max Grid Dimensions: " << prop.maxGridSize[0] << " x " 
         << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << endl;
    
    cudaDeviceGetAttribute(&value, cudaDevAttrMaxBlocksPerMultiprocessor, devId);
    cout << "Max Blocks per SM: " << value << endl;

    cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerMultiProcessor, devId);
    cout << "Max Threads per SM: " << value << endl;
    
    cudaDeviceGetAttribute(&value, cudaDevAttrWarpSize, devId);
    cout << "Warp Size: " << value << endl;
    
    // Register and occupancy info
    cout << endl << "=== Register and Occupancy ===" << endl;
    cout << "Registers per Block: " << prop.regsPerBlock << endl;
    
    cudaDeviceGetAttribute(&value, cudaDevAttrMaxRegistersPerMultiprocessor, devId);
    cout << "Registers per SM: " << value << endl;

    cout << "Multiprocessor Count: " << numProcs << endl;
    cout << "CUDA Cores per SM: ";
    
    // Estimate CUDA cores based on compute capability
    int coresPerSM = 0;
    if (prop.major == 2) coresPerSM = 32;
    else if (prop.major == 3) coresPerSM = 192;
    else if (prop.major == 5) coresPerSM = 128;
    else if (prop.major == 6) coresPerSM = (prop.minor == 1) ? 128 : 64;
    else if (prop.major == 7) coresPerSM = 64;
    else if (prop.major == 8) coresPerSM = (prop.minor == 0) ? 64 : 128;
    else if (prop.major >= 9) coresPerSM = 128;
    
    cout << coresPerSM << endl;
    cout << "Total CUDA Cores: " << coresPerSM * numProcs << endl;
    
    cout << "Base Clock Rate: " << prop.clockRate << " KHz" << endl;

    
    // Performance timing (from your original code)
    cout << endl << "=== Performance Test ===" << endl;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        cudaDeviceGetAttribute(&smemSize, cudaDevAttrMaxSharedMemoryPerBlock, devId);
        cudaDeviceGetAttribute(&numProcs, cudaDevAttrMultiProcessorCount, devId);
    }
    auto end = chrono::high_resolution_clock::now();
    cout << "cudaDeviceGetAttribute average time: "
         << chrono::duration_cast<chrono::microseconds>(end - start).count() / 100.0
         << " us" << endl;
    
    // Additional useful attributes
    cout << endl << "=== Additional Features ===" << endl;
    
    cudaDeviceGetAttribute(&value, cudaDevAttrConcurrentKernels, devId);
    cout << "Concurrent Kernels: " << (value ? "Yes" : "No") << endl;
    
    cudaDeviceGetAttribute(&value, cudaDevAttrAsyncEngineCount, devId);
    cout << "Async Engine Count: " << value << endl;
    
    cudaDeviceGetAttribute(&value, cudaDevAttrUnifiedAddressing, devId);
    cout << "Unified Addressing: " << (value ? "Yes" : "No") << endl;
    
    cudaDeviceGetAttribute(&value, cudaDevAttrManagedMemory, devId);
    cout << "Managed Memory: " << (value ? "Yes" : "No") << endl;
    
    cout << "ECC Enabled: " << (prop.ECCEnabled ? "Yes" : "No") << endl;
    
    return 0;
}