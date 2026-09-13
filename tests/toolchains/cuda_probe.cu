// Exercise a compiled device kernel, not merely CUDA discovery.
#include <cuda_runtime.h>

__global__ void add_one(int* values) {
    values[threadIdx.x] += 1;
}

extern "C" int run_cuda_probe() {
    int input[4] = {3, 7, 11, 19};
    int output[4] = {};
    int* device = nullptr;
    if (cudaMalloc(&device, sizeof(input)) != cudaSuccess) return 1;
    cudaError_t status = cudaMemcpy(device, input, sizeof(input), cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        add_one<<<1, 4>>>(device);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost);
    }
    const cudaError_t released = cudaFree(device);
    if (status != cudaSuccess || released != cudaSuccess) return 2;
    for (int i = 0; i < 4; ++i) {
        if (output[i] != input[i] + 1) return 3;
    }
    return 0;
}
