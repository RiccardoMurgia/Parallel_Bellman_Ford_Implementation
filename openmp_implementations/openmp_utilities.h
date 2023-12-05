//
// Created by rick on 03/12/23.
//

#ifndef PARALLEL_BELLMAN_FORD_IMPLEMENTATION_OPENMP_UTILITIES_H
#define PARALLEL_BELLMAN_FORD_IMPLEMENTATION_OPENMP_UTILITIES_H



typedef struct {
    int value;
    int index;
} MinResult;

void parallel_initialize_distances_0(int *distances, int num_vertices, int source, int maximum_weight);
void parallel_initialize_distances_1(int *distances, int num_vertices, int source, int maximum_weight);
MinResult find_min_value(const int *array, int size);
MinResult parallel_find_min_value_0(const int *array, int size);
MinResult parallel_find_min_value_1(const int *array, int size);



#endif //PARALLEL_BELLMAN_FORD_IMPLEMENTATION_OPENMP_UTILITIES_H
