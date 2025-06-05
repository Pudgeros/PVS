#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

int main() {
    const size_t rows = 1000;
    const size_t cols = 100;
    std::vector<std::vector<float>> A(rows, std::vector<float>(cols));
    std::vector<std::vector<float>> B(rows, std::vector<float>(cols));
    std::vector<std::vector<float>> sum(rows, std::vector<float>(cols));
    std::vector<std::vector<float>> diff(rows, std::vector<float>(cols));
    std::vector<std::vector<float>> prod(rows, std::vector<float>(cols));
    std::vector<std::vector<float>> div(rows, std::vector<float>(cols));

    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j) {
            A[i][j] = static_cast<float>(std::rand() % 100 + 1);
            B[i][j] = static_cast<float>(std::rand() % 100 + 1);
        }

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j) {
            sum[i][j]  = A[i][j] + B[i][j];
            diff[i][j] = A[i][j] - B[i][j];
            prod[i][j] = A[i][j] * B[i][j];
            div[i][j]  = A[i][j] / B[i][j];
        }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << elapsed.count() << std::endl;
    return 0;
}
