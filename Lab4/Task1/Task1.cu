#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Определяем ядро CUDA для вычисления суммы массива
__global__ void reduce_sum(int *input, unsigned long long *result, int size) {
    extern __shared__ unsigned long long cache[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = threadIdx.x;

    unsigned long long temp = 0;
    while (tid < size) {
        temp += input[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[idx] = temp; // Сохраняем промежуточный результат в общей памяти
    __syncthreads(); // Синхронизируем потоки в блоке

    // Суммируем промежуточные результаты в пределах блока
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (idx < s)
            cache[idx] += cache[idx + s]; // Суммируем пары элементов
        __syncthreads(); // Синхронизируем потоки после каждого шага суммирования
    }
    if (idx == 0)
        atomicAdd(result, cache[0]);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <array_size> <threads_per_block>\n", argv[0]);
        return 1;
    }

    int SIZE = atoi(argv[1]);
    int THREADS = atoi(argv[2]);

    if (SIZE <= 0 || THREADS <= 0) {
        printf("Error: array size and threads count must be positive integers\n");
        return 1;
    }

    int *h_array = (int *)malloc(SIZE * sizeof(int)); // Выделяем память под массив на хосте
    int *d_array; // Указатель на массив на устройстве
    unsigned long long *d_result, h_result = 0; // Переменная для хранения результата на хосте и указатель на результат на устройстве

    // Инициализируем массив случайными значениями
    for (int i = 0; i < SIZE; ++i)
        h_array[i] = rand() % 100;

    // Выделяем память на устройстве и копируем данные с хоста на устройство
    cudaMalloc(&d_array, SIZE * sizeof(int));
    cudaMemcpy(d_array, h_array, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Выделяем память под результат на устройстве и инициализируем его нулем
    cudaMalloc(&d_result, sizeof(unsigned long long));
    cudaMemset(d_result, 0, sizeof(unsigned long long));

    // Создаем события для измерения времени выполнения
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Записываем время начала выполнения
    cudaEventRecord(start);

    // Определяем количество блоков
    int blocks = (SIZE + THREADS - 1) / THREADS;
    if (blocks > 128) blocks = 128; // Ограничиваем максимальное количество блоков

    // Запускаем ядро CUDA с динамически выделяемой shared memory
    reduce_sum<<<blocks, THREADS, THREADS * sizeof(unsigned long long)>>>(d_array, d_result, SIZE);

    // Записываем время окончания выполнения и синхронизируемся
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Копируем результат с устройства на хост
    cudaMemcpy(&h_result, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Array size: %d\n", SIZE);
    printf("Threads per block: %d\n", THREADS);
    printf("Blocks: %d\n", blocks);
    printf("CUDA sum = %llu\n", h_result);
    printf("Time: %.6f ms\n", ms);

    // Освобождаем память на устройстве и на хосте
    cudaFree(d_array);
    cudaFree(d_result);
    free(h_array);

    return 0;
}
