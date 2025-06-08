#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int *array = (int *)malloc(n * sizeof(int));
    long long sum = 0;

    if (array == NULL) {
        printf("Failed to allocate memory for the array.\n");
        return 1;
    }

    // Инициализация массива
    for (int i = 0; i < n; i++) {
        array[i] = i + 1;
    }

    // Начало замера времени
    clock_t start_time = clock();

    // Вычисление суммы
    for (int i = 0; i < n; i++) {
        sum += array[i];
    }

    // Конец замера времени
    clock_t end_time = clock();
    double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Sum: %lld\n", sum);
    printf("Time spent: %f seconds\n", time_spent);

    free(array);
    return 0;
}
