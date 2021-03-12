cmake_minimum_required(VERSION 3.0.0)
project(pi_estimation VERSION 0.1.0)

find_package(OpenMP)

add_executable(pi_estimation main.cpp)

target_link_libraries(pi_estimation PRIVATE OpenMP::OpenMP_CXX)