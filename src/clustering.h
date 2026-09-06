#ifndef CLUSTERING_H
#define CLUSTERING_H

#include "postgres.h"
#include "fmgr.h"
#include "vector.h"

/*
 * C function declarations for clustering algorithms
 */
Datum vector_kmeans(PG_FUNCTION_ARGS);
Datum vector_kmeans_assign(PG_FUNCTION_ARGS);
Datum vector_dbscan(PG_FUNCTION_ARGS);
Datum spann_assign(PG_FUNCTION_ARGS);
Datum spann_query(PG_FUNCTION_ARGS);
Datum vector_spectral_clustering(PG_FUNCTION_ARGS);
Datum vector_spectral_project(PG_FUNCTION_ARGS);

#endif							/* CLUSTERING_H */
