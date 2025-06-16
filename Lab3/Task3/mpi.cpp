#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " output_filename.txt\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string filename = argv[1];
    const size_t N = 1000000;

    size_t local_N = N / size;
    size_t start = rank * local_N;
    size_t end = (rank == size - 1) ? N : start + local_N;

    std::vector<float> A_local(end - start), B_local(end - start);
    std::vector<float> sum(end - start), diff(end - start), prod(end - start), div(end - start);

    std::vector<float> A_full, B_full;
    if (rank == 0) {
        A_full.resize(N);
        B_full.resize(N);
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        for (size_t i = 0; i < N; ++i) {
            A_full[i] = static_cast<float>(std::rand() % 100 + 1);
            B_full[i] = static_cast<float>(std::rand() % 100 + 1);
        }
    }

    MPI_Scatter(A_full.data(), local_N, MPI_FLOAT, A_local.data(), local_N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B_full.data(), local_N, MPI_FLOAT, B_local.data(), local_N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < end - start; ++i) {
        sum[i]  = A_local[i] + B_local[i];
        diff[i] = A_local[i] - B_local[i];
        prod[i] = A_local[i] * B_local[i];
        div[i]  = A_local[i] / B_local[i];
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double local_elapsed = std::chrono::duration<double>(end_time - start_time).count();

    double max_time;
    MPI_Reduce(&local_elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::ofstream fout(filename, std::ios::app);
        fout << max_time << std::endl;
        fout.close();
    }

    MPI_Finalize();
    return 0;
}
