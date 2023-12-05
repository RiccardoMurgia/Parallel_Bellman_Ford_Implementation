//
// Created by rick on 25/11/23.
//


#ifndef OPENMP_BELLMAN_FORD_V1_H
#define OPENMP_BELLMAN_FORD_V1_H


#include <stdlib.h>
#include <limits.h>
#include <string.h>

#include "../graph_generator.h"
#include "openmp_utilities.h"



int bellman_ford_v1(Graph *graph, int source, int *dist);


#endif
