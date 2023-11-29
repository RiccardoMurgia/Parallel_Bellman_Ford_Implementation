//
// Created by rick on 25/11/23.
//

#ifndef SERIAL_BELLMAN_FORD_H
#define SERIAL_BELLMAN_FORD_H


#include "graph_generator.h"



void initialize_distances(int* distances, int numVertices, int source, int maximum_weight);
void serial_relax_edges(int* distances, Edge* edges, int numEdges);
int bellman_ford_serial(Graph* graph, int source, int* distances);

#endif