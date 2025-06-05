#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

void quickSort(std::vector<int>& arr, int left, int right) {
    if (left >= right) return;

    int pivot = arr[(left + right) / 2];
    int i = left, j = right;

    while (i <= j) {
        while (arr[i] < pivot) i++;
        while (arr[j] > pivot) j--;
        if (i <= j) {
            std::swap(arr[i], arr[j]);
            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        quickSort(arr, left, j);
        #pragma omp section
        quickSort(arr, i, right);
    }
}

int main() {
    const int N = 1000000;
    std::vector<int> arr(N);

    srand(time(0));
    for (int i = 0; i < N; ++i) {
        arr[i] = rand() % 1000000;
    }

    for (int threads : {2, 4, 8}) {
        omp_set_num_threads(threads);
        std::cout << "=== Threads: " << threads << " ===\n";

        for (int run = 1; run <= 5; ++run) {
            std::vector<int> tempArr = arr;

            auto start = std::chrono::high_resolution_clock::now();
            quickSort(tempArr, 0, N - 1);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Run " << run << ": " << elapsed.count() << " sec\n";
        }
        std::cout << "\n";
    }

    return 0;
}
