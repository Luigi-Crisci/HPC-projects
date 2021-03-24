#include <iostream>
#include <immintrin.h>
#include <chrono>

std::chrono::steady_clock::time_point start, end;
const int size = 131072;
const int ODD_MASK[256] = {1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,
                           1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,
                           1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,
                           1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,
                           1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,
                           1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,
                           1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,
                           1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,
                           1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,
                           1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,
                           1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,
                           1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,
                           1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,
                           1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,
                           1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,
                           1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1};

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

float* sum_odd(float* v1,float* v2, int n){
    float* res = (float*) calloc(n/2,sizeof(float));
    
    for (int i = 1,j=1; i < n; i+=2,j++)
        res[j] = v1[i] + v2[i];
    
    return res;
}

void serial()
{
    float *serial_v1 = get_random_array(size);
    float *serial_v2 = get_random_array(size);

    //Serial computaion
    start = std::chrono::steady_clock::now();
    float *serial_result = sum_odd(serial_v1, serial_v2, size);
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
    __m256i ODD_MASK_VET = _mm256_loadu_si256((__m256i*) ODD_MASK);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < size; i += 256)
    {
        vectorized_v1 = _mm256_maskload_ps(unaligned_array_1 + i, ODD_MASK_VET);
        vectorized_v2 = _mm256_maskload_ps(unaligned_array_2 + i, ODD_MASK_VET);

        result[i / 256] = _mm256_add_ps(vectorized_v1, vectorized_v2);
    }

    end = std::chrono::steady_clock::now();
    std::cout << "Vectorized not aligned: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0 << " ms" << std::endl;

    free(unaligned_array_1);
    free(unaligned_array_2);
}


int main(int, char **)
{
    serial();

    vectorization_not_aligned();

}
