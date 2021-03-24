cmake_minimum_required(VERSION 3.0.0)
project(sum_odd VERSION 0.1.0)

set(CMAKE_CXX_FLAGS "-mavx512f")	

add_executable(sum_odd main.cpp)
