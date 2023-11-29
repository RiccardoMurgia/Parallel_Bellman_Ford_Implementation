//
// Created by rick on 25/11/23.
//


#ifndef OPENMP_BELLMAN_FORD_V1_H
#define OPENMP_BELLMAN_FORD_V1_H

#include "graph_generator.h"
#include "openmp_bellman_ford_V0.h"



typedef struct {
    int value;
    int index;
} MinResult;

MinResult parallel_find_minValue(const int *array, int size);
int bellman_ford_v1(Graph* graph, int source, int* dist);

#endif
