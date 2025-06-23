#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib> // для atoi

using namespace std;
using namespace std::chrono;

__global__ void gpu_array_ops(const float* A, const float* B,
                              float* C_add, float* C_sub,
                              float* C_mul, float* C_div,
                              int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C_add[i] = A[i] + B[i];
        C_sub[i] = A[i] - B[i];
        C_mul[i] = A[i] * B[i];
        C_div[i] = (B[i] != 0) ? (A[i] / B[i]) : 0.0f;
    }
}

int main(int argc, char* argv[]) {
    // Параметры по умолчанию
    int N = 1000000;
    int blockSize = 256;

    // Обработка аргументов командной строки
    if (argc >= 2) {
        N = atoi(argv[1]);
        if (N <= 0) {
            cerr << "Error: Array size must be positive" << endl;
            return 1;
        }
    }
    if (argc >= 3) {
        blockSize = atoi(argv[2]);
        if (blockSize <= 0) {
            cerr << "Error: Block size must be positive" << endl;
            return 1;
        }
    }

    cout << "Using parameters: N = " << N << ", blockSize = " << blockSize << endl;

    float *h_A, *h_B;
    float *h_C_add, *h_C_sub, *h_C_mul, *h_C_div;

    // Выделяем память на хосте
    h_A = new float[N];
    h_B = new float[N];
    h_C_add = new float[N];
    h_C_sub = new float[N];
    h_C_mul = new float[N];
    h_C_div = new float[N];

    // Инициализация массивов случайными числами
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX + 0.01f;
    }

    float *d_A, *d_B, *d_C_add, *d_C_sub, *d_C_mul, *d_C_div;

    // Выделяем память на устройстве
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C_add, N * sizeof(float));
    cudaMalloc(&d_C_sub, N * sizeof(float));
    cudaMalloc(&d_C_mul, N * sizeof(float));
    cudaMalloc(&d_C_div, N * sizeof(float));

    // Копируем данные с хоста на устройство
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    int gridSize = (N + blockSize - 1) / blockSize;

    auto start = high_resolution_clock::now();

    // Запуск kernel
    gpu_array_ops<<<gridSize, blockSize>>>(d_A, d_B, d_C_add, d_C_sub, d_C_mul, d_C_div, N);

    cudaDeviceSynchronize();

    auto end = high_resolution_clock::now();

    // Копируем результат обратно на хост
    cudaMemcpy(h_C_add, d_C_add, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_sub, d_C_sub, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_mul, d_C_mul, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_div, d_C_div, N * sizeof(float), cudaMemcpyDeviceToHost);

    double duration = duration_cast<milliseconds>(end - start).count();

    cout << "GPU time: " << duration << " ms" << endl;

    // Очистка памяти
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_add;
    delete[] h_C_sub;
    delete[] h_C_mul;
    delete[] h_C_div;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_add);
    cudaFree(d_C_sub);
    cudaFree(d_C_mul);
    cudaFree(d_C_div);

    return 0;
}
