#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define SIZE 100000000

int main() {
    int *array = (int *)malloc(SIZE * sizeof(int));
    if (array == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    srand(time(NULL));
    for (int i = 0; i < SIZE; i++) {
        array[i] = rand() % 100;
    }

    
    long long sum = 0;
    double start = omp_get_wtime();

    omp_set_num_threads(6);

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < SIZE; i++) {
        sum += array[i];
    }
    double end = omp_get_wtime();

    double time_spent = end - start;
    printf("Sum: %lld\n", sum);
    printf("Time spent: %f seconds\n", time_spent);

    free(array);
    return 0;
}
