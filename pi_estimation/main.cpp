#include <iostream>
#include <omp.h>
#include <chrono>
#include <errno.h>

double pi_function(double x){
    return 4.0 / (1.0 + (x * x)); 
}

double calculate_p1_sequential(int n){
    const double base = 1.0 / n;
    double height_sum = 0.0;
    double current_x;
    for (int i = 0; i <= n; i++)
    {
        current_x = base * ((double)i);
        height_sum += pi_function(current_x);
    }
    return base * height_sum;
}

double calculate_pi_parallel(int n){
    double area = 0.0;
    const double base = 1.0 / n;
    #pragma omp parallel reduction (+ : area)
    {
        int thread_num = omp_get_thread_num();
        int section_size = n / omp_get_num_threads();
        int start_pos = section_size * thread_num;
        int end_pos = start_pos + section_size;

        double current_x;
        double current_height = 0.0;
        for (int i = start_pos; i < end_pos; i++)
        {
            current_x = base * ((double)i);
            current_height += pi_function(current_x);
        }
        
        area = base * current_height;
    }
    return area;
}

int main(int argc, char** argv) {

    if(argc < 2){
        fprintf(stderr,"Please provide the number of rectangle");
        exit(EXIT_FAILURE);
    }

    int n = strtol(argv[1],NULL,10);
    if(errno == EINVAL){
        fprintf(stderr,"Incorrect parameter: please provide an integer");
        exit(EXIT_FAILURE);
    }

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    double pi = calculate_p1_sequential(n);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    double start_p = omp_get_wtime();
    double pi_p = calculate_pi_parallel(n);
    double stop_p = omp_get_wtime();
    
    std::cout << "Sequential - Pi value: "<<pi<<" - Time: "<< (_Float64)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0 <<" s"<<std::endl;
    std::cout << "Parallel - Pi value: "<<pi_p<<" - Time: "<< stop_p - start_p <<" s"<<std::endl;

    return 0;
}
