#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
    clock_t start = clock();
    for (int i = 0; i < SIZE; i++) {
        sum += array[i];
    }
    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Sum: %lld\n", sum);
    printf("Time spent: %f seconds\n", time_spent);

    free(array);
    return 0;
}
