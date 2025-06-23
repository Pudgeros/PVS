#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void operate_arrays(double **A, double **B, double **result_add, double **result_sub, double **result_mul, double **result_div, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_add[i][j] = A[i][j] + B[i][j];
            result_sub[i][j] = A[i][j] - B[i][j];
            result_mul[i][j] = A[i][j] * B[i][j];
            result_div[i][j] = A[i][j] / B[i][j];
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    int rows = size, cols = size;

    double **A = (double **)malloc(rows * sizeof(double *));
    double **B = (double **)malloc(rows * sizeof(double *));
    double **result_add = (double **)malloc(rows * sizeof(double *));
    double **result_sub = (double **)malloc(rows * sizeof(double *));
    double **result_mul = (double **)malloc(rows * sizeof(double *));
    double **result_div = (double **)malloc(rows * sizeof(double *));

    for (int i = 0; i < rows; i++) {
        A[i] = (double *)malloc(cols * sizeof(double));
        B[i] = (double *)malloc(cols * sizeof(double));
        result_add[i] = (double *)malloc(cols * sizeof(double));
        result_sub[i] = (double *)malloc(cols * sizeof(double));
        result_mul[i] = (double *)malloc(cols * sizeof(double));
        result_div[i] = (double *)malloc(cols * sizeof(double));
    }

    double total_time_spent = 0.0;

    for (int run = 0; run < 100; run++) {
        // Инициализация массивов A и B
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                A[i][j] = i + j;
                B[i][j] = i - j;
            }
        }

        clock_t start = clock(); // Начало измерения времени

        operate_arrays(A, B, result_add, result_sub, result_mul, result_div, rows, cols);

        clock_t end = clock(); // Конец измерения времени
        total_time_spent += (double)(end - start) / CLOCKS_PER_SEC;
    }

    double average_time_spent = total_time_spent / 100;
    printf("Time spent: %f seconds\n", average_time_spent);

    // Освобождение памяти
    for (int i = 0; i < rows; i++) {
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

    return 0;
}
