//
// Created by rick on 03/12/23.
//

#ifndef PARALLEL_BELLMAN_FORD_IMPLEMENTATION_OPENMP_UTILITIES_H
#define PARALLEL_BELLMAN_FORD_IMPLEMENTATION_OPENMP_UTILITIES_H



typedef struct {
    int value;
    int index;
} MinResult;

void parallel_initialize_distances(int *distances, int num_vertices, int source, int maximum_weight);

#endif //PARALLEL_BELLMAN_FORD_IMPLEMENTATION_OPENMP_UTILITIES_H
