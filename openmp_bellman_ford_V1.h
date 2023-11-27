//
// Created by rick on 25/11/23.
//


#ifndef OPENMP_BELLMAN_FORD_V1_H
#define OPENMP_BELLMAN_FORD_V1_H

#include "graph_generator.h"

typedef struct {
    int value;
    int index;
} MinResult;

MinResult findMinValue(const int* array, int size);
void parallel_initialize_distances(int* distances, int numVertices, int source, int maximum_weight);
int bellman_ford_v1(Graph* graph, int source, int* dist);

#endif
