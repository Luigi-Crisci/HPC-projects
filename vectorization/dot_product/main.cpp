#include <iostream>
#include <immintrin.h>
#include <chrono>

std::chrono::steady_clock::time_point start, end;
const int size = 131072; //TODO: why with bigger value of size (2^20 for example) the aligned version is slower?

float *dot_product(float *v1, float *v2, int n)
{

    float *res = (float *)calloc(n, sizeof(float));
    for (int i = 0; i < n; i++)
    {
        res[i] = v1[i] * v2[i];
    }
    return res;
}

void initialize_array(float *v, int n)
{
    for (int i = 0; i < n; i++)
    {
        v[i] = (rand() % 1000) * 0.1;
    }
}

float *get_random_array(int n)
{
    float *res = (float *)calloc(n, sizeof(float));
    initialize_array(res, n);
    return res;
}

void serial()
{
    float *serial_v1 = get_random_array(size);
    float *serial_v2 = get_random_array(size);

    //Serial computaion
    start = std::chrono::steady_clock::now();
    float *serial_result = dot_product(serial_v1, serial_v2, size);
    end = std::chrono::steady_clock::now();

    std::cout << "Serial: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0 << " ms" << std::endl;

    free(serial_v1);
    free(serial_v2);
}

void vectorization_not_aligned()
{
    float *unaligned_array_1 = get_random_array(size);
    float *unaligned_array_2 = get_random_array(size);

    __m256 vectorized_v1, vectorized_v2, result[size / 256];
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < size; i += 256)
    {
        vectorized_v1 = _mm256_loadu_ps(unaligned_array_1 + i);
        vectorized_v2 = _mm256_loadu_ps(unaligned_array_2 + i);

        result[i / 256] = _mm256_mul_ps(vectorized_v1, vectorized_v2);
    }

    end = std::chrono::steady_clock::now();
    std::cout << "Vectorized not aligned: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0 << " ms" << std::endl;

    free(unaligned_array_1);
    free(unaligned_array_2);
}

void vectorization_aligned_data()
{
    float *aligned_array_1 = (float *)aligned_alloc(32, sizeof(float) * size);
    float *aligned_array_2 = (float *)aligned_alloc(32, sizeof(float) * size);

    initialize_array(aligned_array_1, size);
    initialize_array(aligned_array_2, size);

    __m256 vectorized_v1, vectorized_v2, result[size / 256];
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < size; i += 256)
    {
        vectorized_v1 = _mm256_load_ps(aligned_array_1 + i);
        vectorized_v2 = _mm256_load_ps(aligned_array_2 + i);

        result[i / 256] = _mm256_mul_ps(vectorized_v1, vectorized_v2);
    }

    end = std::chrono::steady_clock::now();
    std::cout << "Vectorized aligned: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0 << " ms" << std::endl;

    free(aligned_array_1);
    free(aligned_array_2);
}

int main(int, char **)
{
    serial();

    vectorization_not_aligned();

    vectorization_aligned_data();

}
