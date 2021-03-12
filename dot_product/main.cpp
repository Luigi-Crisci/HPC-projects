#include <iostream>
#include <omp.h>
#include <functional>
#include <chrono>

int sequential_dot_product(int *v1, int *v2, int n)
{
    long int sum = 0;
    for (int i = 0; i < n; i++)
        sum += v1[i] * v2[i];
    return sum;
}

int main(int argc, char **argv)
{
    srand(time(NULL));
    int n = strtol(argv[1], NULL, 10);

    int *v1, *v2;
    v1 = (int *)calloc(n, sizeof(int));
    v2 = (int *)calloc(n, sizeof(int));

    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < n; i++)
        {
            v1[i] = rand_r(&seed) % 3;
            v2[i] = rand_r(&seed) % 3;
        }
    }

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    int sum_sequential = sequential_dot_product(v1, v2, n);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    // reduce
    double start_parallel = omp_get_wtime();
    long int sum = 0;
    #pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++)
        sum += v1[i] * v2[i];
    double stop_parallel = omp_get_wtime();

    std::cout << "Sequential - time elapsed: " << (_Float64)(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) / 1000.0 << " s" << std::endl;
    std::cout << "Parallel - time elapsed: " << stop_parallel - start_parallel << " s" << std::endl;
    return 0;
}
