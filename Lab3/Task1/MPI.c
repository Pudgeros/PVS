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
    double total_time = 0.0;

    if (rank == 0) {
        array = (int*)malloc(n * sizeof(int));
        if (array == NULL) {
            printf("Failed to allocate memory for the array.\n");
            MPI_Finalize();
            return 1;
        }
    }

    int local_n = n / size;
    int* local_array = (int*)malloc(local_n * sizeof(int));
    if (local_array == NULL) {
        printf("Failed to allocate memory for the local array.\n");
        MPI_Finalize();
        return 1;
    }

    // Запускаем цикл 100 раз
    for (int iter = 0; iter < 100; ++iter) {
        if (rank == 0) {
            srand(time(NULL) + iter); // Изменяем seed для rand, чтобы получить разные значения
            for (int i = 0; i < n; i++) {
                array[i] = rand() % 100;
            }
        }

        double start_time, end_time;
        if (rank == 0) {
            start_time = MPI_Wtime();
        }

        MPI_Scatter(array, local_n, MPI_INT, local_array, local_n, MPI_INT, 0, MPI_COMM_WORLD);

        local_sum = 0;
        for (int i = 0; i < local_n; i++) {
            local_sum += local_array[i];
        }

        MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            end_time = MPI_Wtime();
            total_time += end_time - start_time;
        }
    }

    if (rank == 0) {
        printf("Average time spent over 100 runs: %f seconds\n", total_time / 100);
        free(array);
    }

    free(local_array);
    MPI_Finalize();

    return 0;
}
