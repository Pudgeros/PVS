#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>

void bubbleSort(std::vector<int>& array) {
    bool swapped;
    for (size_t i = 0; i < array.size() - 1; ++i) {
        swapped = false;
        for (size_t j = 0; j < array.size() - i - 1; ++j) {
            if (array[j] > array[j + 1]) {
                std::swap(array[j], array[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}

std::vector<int> generateRandomArray(int size, int maxValue = 100) {
    std::vector<int> array(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, maxValue);

    for (auto& num : array) {
        num = distrib(gen);
    }
    return array;
}

int main() {
    const int RUNS = 5;
    const int ARRAY_SIZE = 10000;
    std::vector<long long> runtimes(RUNS);

    for (int run = 0; run < RUNS; ++run) {
        std::vector<int> array = generateRandomArray(ARRAY_SIZE);

        auto start = std::chrono::high_resolution_clock::now();
        bubbleSort(array);
        auto end = std::chrono::high_resolution_clock::now();

        runtimes[run] = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Run " << run + 1 << " time: " << runtimes[run] << " ms\n";
    }

    double average_time = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / RUNS;
    std::cout << "\nAverage time after " << RUNS << " runs: " << average_time << " ms\n";

    return 0;
}
