//
// Created by rick on 28/11/23.
//


#ifndef PARALLEL_BELLMAN_FORD_IMPLEMENTATION_OPENMP_BELLMAN_FORD_V2_H
#define PARALLEL_BELLMAN_FORD_IMPLEMENTATION_OPENMP_BELLMAN_FORD_V2_H


#include <stdlib.h>
#include <limits.h>
#include <string.h>

#include "../graph_generator.h"
#include "openmp_utilities.h"
#include "omp.h"


int bellman_ford_v2(Graph *graph, int source, int *dist, int *predecessor);


#endif //PARALLEL_BELLMAN_FORD_IMPLEMENTATION_OPENMP_BELLMAN_FORD_V2_H
