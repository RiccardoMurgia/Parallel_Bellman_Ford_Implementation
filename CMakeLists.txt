cmake_minimum_required(VERSION 3.26)
project(Parallel_Bellman_Ford_Implementation)

set(CMAKE_C_STANDARD 23)

add_executable(Parallel_Bellman_Ford_Implementation
               main.c
               openmp_bellman_ford_V0.c
               graph_generator.c
               openmp_bellman_ford_V1.c
        bellman_ford_Sq.c
               openmp_bellman_ford_V2.c
               openmp_bellman_ford_V2.h
        utilities.c
        utilities.h)

find_package(OpenMP)

if (OpenMP_CXX_FOUND)
    target_compile_options(Parallel_Bellman_Ford_Implementation PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(Parallel_Bellman_Ford_Implementation PRIVATE ${OpenMP_CXX_LIBRARIES} m)
endif()
