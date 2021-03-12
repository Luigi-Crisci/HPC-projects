cmake_minimum_required(VERSION 3.0.0)
project(dot_product VERSION 0.1.0)

find_package(OpenMP REQUIRED)

add_executable(dot_product main.cpp)

target_link_libraries(dot_product PRIVATE OpenMP::OpenMP_CXX)
