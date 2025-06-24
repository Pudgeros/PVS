#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Функция для слияния двух подмассивов
void merge(float arr[], int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    // Создаем временные массивы
    float L[n1], R[n2];

    // Копируем данные во временные массивы
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    // Слияние временных массивов обратно в arr[l..r]
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Копируем оставшиеся элементы L[], если они есть
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Копируем оставшиеся элементы R[], если они есть
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// Рекурсивная функция сортировки слиянием
void mergeSort(float arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        // Сортируем первую и вторую половины
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        merge(arr, l, m, r);
    }
}

// Функция для генерации случайного массива
void generateRandomArray(float arr[], int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)rand() / RAND_MAX * 1000.0f;
    }
}

int main() {
    // Размеры массивов для тестирования
    int sizes[] = {100000, 500000, 1000000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i];
        float *arr = (float *)malloc(n * sizeof(float));
        
        // Генерация случайного массива
        generateRandomArray(arr, n);
        
        // Замер времени
        clock_t start = clock();
        mergeSort(arr, 0, n - 1);
        clock_t end = clock();
        
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
        printf("Merge Sort for %d elements: %.3f seconds\n", n, time_spent);
        
        free(arr);
    }

    return 0;
}
