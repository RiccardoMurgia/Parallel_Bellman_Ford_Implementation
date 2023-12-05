//
// Created by rick on 25/11/23.
//


#ifndef OPENMP_BELLMAN_FORD_V0_H
#define OPENMP_BELLMAN_FORD_V0_H


#include "../graph_generator.h"
#include "openmp_utilities.h"


void parallel_relax_edges_0(int *distances, Edge *edges, int num_edges);
int bellman_ford_v0(Graph *graph, int source, int *distances);


#endif
