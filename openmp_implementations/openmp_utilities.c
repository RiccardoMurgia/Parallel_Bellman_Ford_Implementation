//
// Created by rick on 03/12/23.
//


#include <limits.h>
#include "openmp_utilities.h"



void parallel_initialize_distances_0(int *distances, int num_vertices, int source, int maximum_weight){
    #pragma omp parallel for default(none)  firstprivate(distances, num_vertices, source, maximum_weight)
        for (int i = 0; i < num_vertices; i++)
            distances[i] = (i == source) ? 0 : INT_MAX - maximum_weight;
}


void parallel_initialize_distances_1(int *distances, int num_vertices, int source, int maximum_weight){
    #pragma omp for
        for (int i = 0; i < num_vertices; i++)
            distances[i] = (i == source) ? 0 : INT_MAX - maximum_weight;

}


MinResult find_min_value(const int *array, int size){
    MinResult result;
    result.value = INT_MAX;
    result.index = -1;

    for (int i = 0; i < size; i++){
        if (array[i] < result.value) {
            result.value = array[i];
            result.index = i;
        }
    }

    return result;
}


MinResult parallel_find_min_value(const int *array, int size){
    MinResult result;
    result.value = INT_MAX;
    result.index = -1;

    #pragma omp declare reduction(MinResultMin: MinResult: omp_out = omp_in.value < omp_out.value ? omp_in : omp_out) initializer(omp_priv = {INT_MAX, -1})

    #pragma omp parallel for default(none) firstprivate(array, size) reduction(MinResultMin: result) num_threads(1)
        for (int i = 0; i < size; i++){
            if (array[i] < result.value){
                result.value = array[i];
                result.index = i;
            }
        }

    return result;
}



