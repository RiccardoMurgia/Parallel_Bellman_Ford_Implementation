// cuda_functions.cu

#include <cstdio>

// CUDA function definition
__global__ void kernel() {
    printf("Hello from CUDA! ThreadIdx: %d\n", threadIdx.x);
}

extern "C" void helloCUDA() {
    // Launch the CUDA kernel
    kernel<<<1, 10>>>();
    cudaDeviceSynchronize();
}
