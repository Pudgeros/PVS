#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <cstdlib>

using namespace std;

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error в " << __FILE__ << ":" << __LINE__ << ": " \
                 << cudaGetErrorString(err) << endl; \
            exit(EXIT_FAILURE); \
        } \
    }

// CUDA kernel для операций над элементами массивов
__global__ void gpu_array_ops(const float* A, const float* B,
                              float* C_add, float* C_sub,
                              float* C_mul, float* C_div,
                              int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C_add[idx] = A[idx] + B[idx];
        C_sub[idx] = A[idx] - B[idx];
        C_mul[idx] = A[idx] * B[idx];
        C_div[idx] = (B[idx] != 0.0f) ? A[idx] / B[idx] : 0.0f;
    }
}

// Последовательная реализация на CPU
void cpu_array_ops(const float* A, const float* B,
                   float* C_add, float* C_sub,
                   float* C_mul, float* C_div,
                   int size) {
    for (int i = 0; i < size; i++) {
        C_add[i] = A[i] + B[i];
        C_sub[i] = A[i] - B[i];
        C_mul[i] = A[i] * B[i];
        C_div[i] = (B[i] != 0.0f) ? A[i] / B[i] : 0.0f;
    }
}

void print_usage(const char* prog_name) {
    cerr << "Использование: " << prog_name << " [M N] [block_size]" << endl;
    cerr << "  M - размер по первому измерению (по умолчанию 500)" << endl;
    cerr << "  N - размер по второму измерению (по умолчанию 200)" << endl;
    cerr << "  block_size - размер блока CUDA (по умолчанию 256)" << endl;
}

int main(int argc, char* argv[]) {
    // Параметры по умолчанию
    int M = 500;      // Размер массива по первому измерению
    int N = 200;      // Размер массива по второму измерению
    int block_size = 256; // Размер блока CUDA

    // Обработка аргументов командной строки
    if (argc >= 3) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        if (M <= 0 || N <= 0) {
            cerr << "Ошибка: размеры массива должны быть положительными" << endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (argc >= 4) {
        block_size = atoi(argv[3]);
        if (block_size <= 0 || block_size > 1024) {
            cerr << "Ошибка: размер блока должен быть между 1 и 1024" << endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    const int size = M * N; // Общее количество элементов

    cout << "Размеры массива: " << M << " x " << N << " = " << size << " элементов" << endl;
    cout << "Размер блока CUDA: " << block_size << endl;

    // Выделяем память на CPU
    float *h_A = new float[size];
    float *h_B = new float[size];
    float *h_C_add_cpu = new float[size];
    float *h_C_sub_cpu = new float[size];
    float *h_C_mul_cpu = new float[size];
    float *h_C_div_cpu = new float[size];

    float *h_C_add_gpu = new float[size];
    float *h_C_sub_gpu = new float[size];
    float *h_C_mul_gpu = new float[size];
    float *h_C_div_gpu = new float[size];

    // Инициализируем массивы случайными числами (сдвиг чтобы избежать нуля)
    srand(0);
    for (int i = 0; i < size; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX + 0.01f;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX + 0.01f;
    }

    // Последовательное выполнение на CPU и измерение времени
    auto cpu_start = chrono::high_resolution_clock::now();
    cpu_array_ops(h_A, h_B, h_C_add_cpu, h_C_sub_cpu, h_C_mul_cpu, h_C_div_cpu, size);
    auto cpu_end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> cpu_duration = cpu_end - cpu_start;
    cout << "Время выполнения CPU: " << cpu_duration.count() << " мс" << endl;

    // Выделение памяти на GPU
    float *d_A, *d_B, *d_C_add, *d_C_sub, *d_C_mul, *d_C_div;
    CUDA_CHECK(cudaMalloc(&d_A, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_add, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_sub, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_mul, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_div, size * sizeof(float)));

    // Копируем данные с CPU на GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice));

    // Создаем события для измерения времени выполнения на GPU
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Вычисляем количество блоков
    int grid_size = (size + block_size - 1) / block_size;

    // Запускаем ядро CUDA и измеряем время выполнения
    CUDA_CHECK(cudaEventRecord(start));
    gpu_array_ops<<<grid_size, block_size>>>(d_A, d_B, d_C_add, d_C_sub, d_C_mul, d_C_div, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    cout << "Время выполнения GPU kernel: " << gpu_time_ms << " мс" << endl;

    // Копируем результаты с GPU обратно на CPU
    CUDA_CHECK(cudaMemcpy(h_C_add_gpu, d_C_add, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_sub_gpu, d_C_sub, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_mul_gpu, d_C_mul, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_div_gpu, d_C_div, size * sizeof(float), cudaMemcpyDeviceToHost));

    // Проверяем корректность (сравним первые 10 элементов)
    cout << "Первые 10 результатов (CPU vs GPU) для сложения:" << endl;
    for (int i = 0; i < 10; i++) {
        cout << i << ": CPU=" << h_C_add_cpu[i] << ", GPU=" << h_C_add_gpu[i] << endl;
    }

    // Сохраняем первые 20 результатов сложения в файл
    ofstream fout("results.txt");
    if (!fout) {
        cerr << "Ошибка при открытии файла results.txt" << endl;
    } else {
        for (int i = 0; i < 20; i++) {
            fout << h_C_add_gpu[i] << " " << h_C_sub_gpu[i] << " "
                 << h_C_mul_gpu[i] << " " << h_C_div_gpu[i] << "\n";
        }
        fout.close();
    }

    // Освобождаем память
    delete[] h_A; delete[] h_B;
    delete[] h_C_add_cpu; delete[] h_C_sub_cpu; delete[] h_C_mul_cpu; delete[] h_C_div_cpu;
    delete[] h_C_add_gpu; delete[] h_C_sub_gpu; delete[] h_C_mul_gpu; delete[] h_C_div_gpu;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_add));
    CUDA_CHECK(cudaFree(d_C_sub));
    CUDA_CHECK(cudaFree(d_C_mul));
    CUDA_CHECK(cudaFree(d_C_div));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
