//
// Created by rick on 04/12/23.
//

#ifndef PARALLEL_BELLMAN_FORD_IMPLEMENTATION_OPENMP_BELLMAN_FORD_V01_H
#define PARALLEL_BELLMAN_FORD_IMPLEMENTATION_OPENMP_BELLMAN_FORD_V01_H


#include "../graph_generator.h"
#include "openmp_utilities.h"


void parallel_relax_edges_1(int *distances, int *predecessor, Edge *edges, int num_edges);
int bellman_ford_v0_1(Graph *graph, int source, int *distances, int *predecessor);

#endif //PARALLEL_BELLMAN_FORD_IMPLEMENTATION_OPENMP_BELLMAN_FORD_V01_H
