//
// Created by rick on 03/12/23.
//


#include <limits.h>
#include "openmp_utilities.h"



void parallel_initialize_distances(int *distances, int num_vertices, int source, int maximum_weight){
#pragma omp parallel for default(none)  firstprivate(distances, num_vertices, source, maximum_weight)
    for (int i = 0; i < num_vertices; i++)
        distances[i] = (i == source) ? 0 : INT_MAX - maximum_weight;

}
