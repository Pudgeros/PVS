#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// Ядро для сравнения и обмена элементов
__global__ void bitonicSortStep(float *devValues, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;
    
    if (ixj > i) {
        if ((i & k) == 0) {
            // Сортировка по возрастанию
            if (devValues[i] > devValues[ixj]) {
                float temp = devValues[i];
                devValues[i] = devValues[ixj];
                devValues[ixj] = temp;
            }
        } else {
            // Сортировка по убыванию
            if (devValues[i] < devValues[ixj]) {
                float temp = devValues[i];
                devValues[i] = devValues[ixj];
                devValues[ixj] = temp;
            }
        }
    }
}

// Функция для битонической сортировки на GPU
void bitonicSort(float *values, int n) {
    float *devValues;
    size_t size = n * sizeof(float);
    
    // Выделяем память на GPU
    cudaMalloc((void**)&devValues, size);
    cudaMemcpy(devValues, values, size, cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    // Основной алгоритм битонической сортировки
    for (int k = 2; k <= n; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            bitonicSortStep<<<blocks, threads>>>(devValues, j, k);
        }
    }
    
    // Копируем результат обратно на CPU
    cudaMemcpy(values, devValues, size, cudaMemcpyDeviceToHost);
    cudaFree(devValues);
}

// Функция для проверки правильности сортировки
int isSorted(float *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return 0;
        }
    }
    return 1;
}

int main() {
    // Размеры массивов для тестирования
    int sizes[] = {100000, 500000, 1000000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i];
        // Убедимся, что размер является степенью двойки (требование для Bitonic Sort)
        int pow2 = 1;
        while (pow2 < n) pow2 *= 2;
        n = pow2;
        
        float *arr = (float *)malloc(n * sizeof(float));
        
        // Генерация случайного массива
        for (int j = 0; j < n; j++) {
            arr[j] = (float)rand() / RAND_MAX * 1000.0f;
        }
        
        // Замер времени
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        bitonicSort(arr, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        double seconds = milliseconds / 1000.0;
        
        printf("Bitonic Sort for %d elements: %.3f seconds\n", n, seconds);
        
        // Проверка правильности сортировки
        if (isSorted(arr, n)) {
            printf("Array is correctly sorted.\n");
        } else {
            printf("Array is NOT sorted correctly!\n");
        }
        
        free(arr);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}
