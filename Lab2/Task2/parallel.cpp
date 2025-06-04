#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <omp.h>

using namespace std;

void parallel_quicksort(vector<int>& arr, size_t left, size_t right) {
    if (left >= right) return;

    int pivot = arr[(left + right) / 2];
    size_t i = left, j = right;

    while (i <= j) {
        while (arr[i] < pivot) i++;
        while (arr[j] > pivot) j--;
        if (i <= j) {
            swap(arr[i], arr[j]);
            i++;
            if (j > 0) j--;
        }
    }

    #pragma omp task shared(arr)
    if (j > left) parallel_quicksort(arr, left, j);

    #pragma omp task shared(arr)
    if (i < right) parallel_quicksort(arr, i, right);

    #pragma omp taskwait
}

vector<int> generate_random_array(size_t size) {
    vector<int> arr(size);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 10000000);

    for (size_t i = 0; i < size; ++i) {
        arr[i] = dis(gen);
    }
    return arr;
}

int main() {
    constexpr size_t ARRAY_SIZE = 10000000;
    constexpr int NUM_RUNS = 5;
    auto arr = generate_random_array(ARRAY_SIZE);

    for (int threads : {2, 4, 8}) {
        omp_set_num_threads(threads);
        cout << "Threads: " << threads << endl;

        for (int i = 0; i < NUM_RUNS; ++i) {
            auto arr_copy = arr;
            auto start = chrono::high_resolution_clock::now();

            #pragma omp parallel
            {
                #pragma omp single
                parallel_quicksort(arr_copy, 0, arr_copy.size() - 1);
            }

            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = end - start;

            cout << "Run " << i+1 << ": " << elapsed.count() << "s" << endl;
        }
    }

    return 0;
}
