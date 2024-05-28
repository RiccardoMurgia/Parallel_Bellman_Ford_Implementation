if [ ! -d "obj" ]; then
    mkdir obj
fi

gcc -c -fopenmp main.c -o obj/main.o;
gcc -c -fopenmp general_utilities.c -o obj/general_utilities.o
gcc -c -fopenmp graph_generator.c -o obj/graph_generator.o
gcc -c -fopenmp bellman_ford_Sq.c -o obj/bellman_ford_Sq.o

gcc -c -fopenmp openmp_implementations/openmp_utilities.c -o obj/openmp_utilities.o
gcc -c -fopenmp openmp_implementations/openmp_bellman_ford_V0.c -o obj/openmp_bellman_ford_V0.o
gcc -c -fopenmp openmp_implementations/openmp_bellman_ford_V0_1.c -o obj/openmp_bellman_ford_V0_1.o
gcc -c -fopenmp openmp_implementations/openmp_bellman_ford_V1.c -o obj/openmp_bellman_ford_V1.o
gcc -c -fopenmp openmp_implementations/openmp_bellman_ford_V1_1.c -o obj/openmp_bellman_ford_V1_1.o
gcc -c -fopenmp openmp_implementations/openmp_bellman_ford_V2.c -o obj/openmp_bellman_ford_V2.o
gcc -c -fopenmp openmp_implementations/openmp_bellman_ford_V2_1.c -o obj/openmp_bellman_ford_V2_1.o

nvcc -c cuda_implementations/cuda_utilities.cu -o obj/cuda_utilities.o -arch=sm_80
nvcc -c cuda_implementations/cuda_bellman_ford_V0.cu -o obj/cuda_bellman_ford_V0.o -arch=sm_80
nvcc -c cuda_implementations/cuda_bellman_ford_V0_1.cu -o obj/cuda_bellman_ford_V0_1.o -arch=sm_80
nvcc -c cuda_implementations/cuda_bellman_ford_V1.cu -o obj/cuda_bellman_ford_V1.o -arch=sm_80
nvcc -c cuda_implementations/cuda_bellman_ford_V1_1.cu -o obj/cuda_bellman_ford_V1_1.o -arch=sm_80

gcc -fopenmp obj/general_utilities.o obj/graph_generator.o obj/bellman_ford_Sq.o \
    obj/openmp_utilities.o obj/openmp_bellman_ford_V0.o obj/openmp_bellman_ford_V0_1.o obj/openmp_bellman_ford_V1.o \
    obj/openmp_bellman_ford_V1_1.o obj/openmp_bellman_ford_V2.o obj/openmp_bellman_ford_V2_1.o \
    obj/cuda_utilities.o obj/cuda_bellman_ford_V0.o obj/cuda_bellman_ford_V0_1.o obj/cuda_bellman_ford_V1.o  \
    obj/cuda_bellman_ford_V1_1.o obj/main.o -o executable -lm -L/opt/cuda/lib64 -lcudart

./executable 0 0 0 1
