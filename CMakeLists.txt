#[[
gcc -c -fopenmp main.c -o main.o;
gcc -c -fopenmp utilities.c -o utilities.o;
gcc -c -fopenmp graph_generator.c -o graph_generator.o;
gcc -c -fopenmp bellman_ford_Sq.c -o bellman_ford_Sq.o;
gcc -c -fopenmp openmp_bellman_ford_V0.c -o openmp_bellman_ford_V0.o;
gcc -c -fopenmp openmp_bellman_ford_V1.c -o openmp_bellman_ford_V1.o;
gcc -c -fopenmp openmp_bellman_ford_V2.c -o openmp_bellman_ford_V2.o;
nvcc -c cuda_version.cu -o cuda_version.o -arch=sm_80;
gcc -fopenmp utilities.o graph_generator.o bellman_ford_Sq.o openmp_bellman_ford_V0.o openmp_bellman_ford_V1.o openmp_bellman_ford_V2.o cuda_version.o main.o -o executable -lm -L/opt/cuda/lib64 -lcudart
]]


cmake_minimum_required(VERSION 3.26)
project(Parallel_Bellman_Ford_Implementation C CUDA)

# Setting cuda version
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Setting C version
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Adding source files
set(SOURCES
        general_utilities.c
        graph_generator.c

        bellman_ford_Sq.c

        openmp_implementations/openmp_utilities.c
        openmp_implementations/openmp_bellman_ford_V0.c
        openmp_implementations/openmp_bellman_ford_V0_1.c
        openmp_implementations/openmp_bellman_ford_V1.c
        openmp_implementations/openmp_bellman_ford_V1_1.c
        openmp_implementations/openmp_bellman_ford_V2.c
        openmp_implementations/openmp_bellman_ford_V2_1.c

        cuda_implementations/cuda_utilities.cu
        cuda_implementations/cuda_bellman_ford_V0.cu
        cuda_implementations/cuda_bellman_ford_V0_1.cu
        cuda_implementations/cuda_bellman_ford_V1.cu
        cuda_implementations/cuda_bellman_ford_V1_1.cu

        main.c
)

# Adding compilation specifics
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")

# Setting executable file
add_executable(Parallel_Bellman_Ford_Implementation ${SOURCES})

# Linking libraries
find_package(OpenMP REQUIRED)
target_link_libraries(Parallel_Bellman_Ford_Implementation PRIVATE OpenMP::OpenMP_C)
target_link_libraries(Parallel_Bellman_Ford_Implementation PRIVATE m)
target_link_libraries(Parallel_Bellman_Ford_Implementation PRIVATE cudart)

# Specify CUDA libraries directory
link_directories(/opt/cuda/lib64)
