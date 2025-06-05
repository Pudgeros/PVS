#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

int main() {
    const size_t N = 100000000;
    std::vector<float> A(N), B(N), sum(N), diff(N), prod(N), div(N);

    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (size_t i = 0; i < N; ++i) {
        A[i] = static_cast<float>(std::rand() % 100 + 1);
        B[i] = static_cast<float>(std::rand() % 100 + 1);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < N; ++i) {
        sum[i]  = A[i] + B[i];
        diff[i] = A[i] - B[i];
        prod[i] = A[i] * B[i];
        div[i]  = A[i] / B[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << elapsed.count() << std::endl;
    return 0;
}
