#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>
#include <stdexcept>

void parallel_bubble_sort(std::vector<int>& data) {
    bool swapped;
    do {
        swapped = false;
        for (size_t i = 0; i < data.size() - 1; ++i) {
            if (data[i] > data[i + 1]) {
                std::swap(data[i], data[i + 1]);
                swapped = true;
            }
        }
    } while (swapped);
}

int main(int argc, char** argv) {
    try {
        MPI_Init(&argc, &argv);

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        const int RUNS = 5;
        const size_t N = (argc > 1) ? std::stoul(argv[1]) : 10000;
        if (N % size != 0) {
            if (rank == 0) {
                std::cerr << "Error: Array size must be divisible by the number of processes\n";
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        const size_t local_size = N / size;
        std::vector<long long> runtimes(RUNS);

        for (int run = 0; run < RUNS; ++run) {
            std::vector<int> global_data;
            std::vector<int> local_data(local_size);

            MPI_Barrier(MPI_COMM_WORLD);
            auto start_time = std::chrono::high_resolution_clock::now();

            if (rank == 0) {
                global_data.resize(N);
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> distrib(0, 99);

                for (auto& num : global_data) {
                    num = distrib(gen);
                }
            }

            MPI_Scatter(global_data.data(), local_size, MPI_INT,
                      local_data.data(), local_size, MPI_INT,
                      0, MPI_COMM_WORLD);

            parallel_bubble_sort(local_data);

            MPI_Gather(local_data.data(), local_size, MPI_INT,
                     global_data.data(), local_size, MPI_INT,
                     0, MPI_COMM_WORLD);

            if (rank == 0) {
                parallel_bubble_sort(global_data);
                auto end_time = std::chrono::high_resolution_clock::now();
                runtimes[run] = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
                std::cout << "Run " << run + 1 << " time: " << runtimes[run] << " ms\n";
            }
        }

        if (rank == 0) {
            double average_time = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / RUNS;
            std::cout << "\nAverage time after " << RUNS << " runs: " << average_time << " ms\n";
            std::cout << "Array size: " << N << " elements\n";
            std::cout << "Number of processes: " << size << "\n";
        }

        MPI_Finalize();
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
}
