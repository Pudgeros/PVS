#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

void operate_arrays(double** A, double** B, double** result_add, double** result_sub, double** result_mul, double** result_div, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_add[i][j] = A[i][j] + B[i][j];
            result_sub[i][j] = A[i][j] - B[i][j];
            result_mul[i][j] = A[i][j] * B[i][j];
            result_div[i][j] = A[i][j] / B[i][j];
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: mpirun -np <num_processes> %s <array_size>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int array_size = atoi(argv[1]);
    int rows = array_size, cols = array_size;
    int local_rows = rows / size;

    double** A = (double**)malloc(local_rows * sizeof(double*));
    double** B = (double**)malloc(local_rows * sizeof(double*));
    double** result_add = (double**)malloc(local_rows * sizeof(double*));
    double** result_sub = (double**)malloc(local_rows * sizeof(double*));
    double** result_mul = (double**)malloc(local_rows * sizeof(double*));
    double** result_div = (double**)malloc(local_rows * sizeof(double*));

    for (int i = 0; i < local_rows; i++) {
        A[i] = (double*)malloc(cols * sizeof(double));
        B[i] = (double*)malloc(cols * sizeof(double));
        result_add[i] = (double*)malloc(cols * sizeof(double));
        result_sub[i] = (double*)malloc(cols * sizeof(double));
        result_mul[i] = (double*)malloc(cols * sizeof(double));
        result_div[i] = (double*)malloc(cols * sizeof(double));
    }

    double total_time_spent = 0.0;

    for (int run = 0; run < 100; run++) {
        // Инициализация массивов A и B
        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < cols; j++) {
                A[i][j] = rank * local_rows + i + j;
                B[i][j] = rank * local_rows + i - j;
            }
        }

        double start_time, end_time;
        MPI_Barrier(MPI_COMM_WORLD); // Синхронизация перед началом измерения времени
        start_time = MPI_Wtime(); // Начало измерения времени

        operate_arrays(A, B, result_add, result_sub, result_mul, result_div, local_rows, cols);

        MPI_Barrier(MPI_COMM_WORLD); // Синхронизация перед окончанием измерения времени
        end_time = MPI_Wtime(); // Конец измерения времени

        total_time_spent += end_time - start_time;
    }

    double average_time_spent = total_time_spent / 100;

    if (rank == 0) {
        printf("Среднее время выполнения: %f секунд\n", average_time_spent);
    }

    // Освобождение памяти
    for (int i = 0; i < local_rows; i++) {
        free(A[i]);
        free(B[i]);
        free(result_add[i]);
        free(result_sub[i]);
        free(result_mul[i]);
        free(result_div[i]);
    }
    free(A);
    free(B);
    free(result_add);
    free(result_sub);
    free(result_mul);
    free(result_div);

    MPI_Finalize();
    return 0;
}
