#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s <array_size>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int n = atoi(argv[1]);
    int* array = NULL;
    long long local_sum = 0;
    long long global_sum = 0;

    if (rank == 0) {
        array = (int*)malloc(n * sizeof(int));
        if (array == NULL) {
            printf("Failed to allocate memory for the array.\n");
            MPI_Finalize();
            return 1;
        }
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            array[i] = rand() % 100;
        }
    }

    // Разбиваем массив на части для каждого процесса
    int local_n = n / size;
    int* local_array = (int*)malloc(local_n * sizeof(int));
    if (local_array == NULL) {
        printf("Failed to allocate memory for the local array.\n");
        MPI_Finalize();
        return 1;
    }

    // Начало замера времени
    double start_time, end_time;
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    // Распределяем части массива между процессами
    MPI_Scatter(array, local_n, MPI_INT, local_array, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Локальное вычисление суммы
    for (int i = 0; i < local_n; i++) {
        local_sum += local_array[i];
    }

    // Собираем результаты на процесс 0
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Конец замера времени
    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("Sum: %lld\n", global_sum);
        printf("Time spent: %f seconds\n", end_time - start_time);
        free(array);
    }

    free(local_array);
    MPI_Finalize();

    return 0;
}
