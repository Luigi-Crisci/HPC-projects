cmake_minimum_required(VERSION 3.0.0)
project(vectorization_sum VERSION 0.1.0)

set(CMAKE_CXX_FLAGS "-mavx")	

add_executable(vectorization_sum main.cpp)
