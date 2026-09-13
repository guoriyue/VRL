#include <cstdio>

extern "C" int run_cuda_probe();

int main() {
    const int result = run_cuda_probe();
    if (result != 0) std::fprintf(stderr, "CUDA execution failed: %d\n", result);
    return result;
}
