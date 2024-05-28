//
// Created by rick on 04/12/23.
//


#ifndef PARALLEL_BELLMAN_FORD_IMPLEMENTATION_OPENMP_BELLMAN_FORD_V1_1_H
#define PARALLEL_BELLMAN_FORD_IMPLEMENTATION_OPENMP_BELLMAN_FORD_V1_1_H


#include <stdlib.h>
#include <limits.h>
#include <string.h>

#include "../graph_generator.h"
#include "openmp_utilities.h"



int bellman_ford_v1_1(Graph *graph, int source, int *dist, int *predecessor);


#endif //PARALLEL_BELLMAN_FORD_IMPLEMENTATION_OPENMP_BELLMAN_FORD_V1_1_H
