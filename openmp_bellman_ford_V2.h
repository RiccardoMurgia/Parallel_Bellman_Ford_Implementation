//
// Created by rick on 28/11/23.
//

#ifndef PARALLEL_BELLMAN_FORD_IMPLEMENTATION_OPENMP_BELLMAN_FORD_V2_H
#define PARALLEL_BELLMAN_FORD_IMPLEMENTATION_OPENMP_BELLMAN_FORD_V2_H


#include "openmp_bellman_ford_V1.h"
#include "graph_generator.h"



MinResult parallel_find_min_value(const int* array, int size);
int bellman_ford_v2(Graph* graph, int source, int* dist);


#endif //PARALLEL_BELLMAN_FORD_IMPLEMENTATION_OPENMP_BELLMAN_FORD_V2_H
