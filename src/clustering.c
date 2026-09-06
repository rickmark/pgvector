#include "postgres.h"

#include <float.h>
#include <limits.h>
#include <math.h>

#include "catalog/pg_type.h"
#include "clustering.h"
#include "fmgr.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "vector.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#if PG_VERSION_NUM >= 190000
#define palloc_array_checked(type, count) ((type *) palloc_array(type, count))
#else
#define palloc_array_checked(type, count) ((type *) palloc(mul_size(sizeof(type), count)))
#endif

/*
 * Distance calculation helpers
 */
static inline float
VectorDistSq(int dim, const float *a, const float *b)
{
	float		distance = 0.0f;

	for (int i = 0; i < dim; i++)
	{
		float		diff = a[i] - b[i];

		distance += diff * diff;
	}
	return distance;
}

static inline float
VectorDistL2(int dim, const float *a, const float *b)
{
	return sqrtf(VectorDistSq(dim, a, b));
}

/*
 * Extract array of Vector* pointers from PostgreSQL 1-D Vector Array
 */
static Vector **
ExtractVectorArray(ArrayType *array, int *num_vectors, int *dim)
{
	int			ndims = ARR_NDIM(array);
	int		   *dims = ARR_DIMS(array);
	int			nitems;
	Datum	   *datums;
	bool	   *nulls;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	Vector	  **vectors;

	if (ndims > 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("array must be 1-D")));

	if (ARR_HASNULL(array) && array_contains_nulls(array))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("array must not contain nulls")));

	nitems = ArrayGetNItems(ndims, dims);
	if (nitems == 0)
	{
		*num_vectors = 0;
		*dim = 0;
		return NULL;
	}

	get_typlenbyvalalign(ARR_ELEMTYPE(array), &typlen, &typbyval, &typalign);
	deconstruct_array(array, ARR_ELEMTYPE(array), typlen, typbyval, typalign, &datums, &nulls, &nitems);

	vectors = palloc_array_checked(Vector *, (Size) nitems);
	for (int i = 0; i < nitems; i++)
	{
		vectors[i] = DatumGetVector(datums[i]);
		if (i == 0)
			*dim = vectors[0]->dim;
		else if (vectors[i]->dim != *dim)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("all vectors in array must have the same dimension: expected %d, got %d at index %d",
							*dim, vectors[i]->dim, i + 1)));
	}

	*num_vectors = nitems;
	pfree(datums);
	pfree(nulls);
	return vectors;
}

/*
 * Internal k-Means implementation on array of Vector pointers
 */
static void
RunKMeansInternal(Vector **vectors, int n, int dim, int k, int max_iter, Vector **centers, int *assignments)
{
	float	   *weights = palloc_array_checked(float, (Size) n);
	int		   *cluster_counts = palloc_array_checked(int, (Size) k);
	float	   *cluster_sums = palloc_array_checked(float, (Size) k * (Size) dim);
	double		movement;

	/* Step 1: k-means++ initialization */
	centers[0] = InitVector(dim);
	for (int d = 0; d < dim; d++)
		centers[0]->x[d] = vectors[0]->x[d];

	for (int i = 0; i < n; i++)
		weights[i] = FLT_MAX;

	for (int c = 0; c < k - 1; c++)
	{
		double		sum = 0.0;
		double		choice;
		int			selected_idx = 0;

		CHECK_FOR_INTERRUPTS();

		for (int i = 0; i < n; i++)
		{
			float		d = VectorDistSq(dim, vectors[i]->x, centers[c]->x);

			if (d < weights[i])
				weights[i] = d;
			sum += weights[i];
		}

		if (sum <= 0.0)
		{
			selected_idx = (c + 1) % n;
		}
		else
		{
			/* Deterministic-proportional or random choice */
			choice = sum * ((double) rand() / ((double) RAND_MAX + 1.0));
			for (int i = 0; i < n; i++)
			{
				choice -= weights[i];
				if (choice <= 0.0 || i == n - 1)
				{
					selected_idx = i;
					break;
				}
			}
		}

		centers[c + 1] = InitVector(dim);
		for (int d = 0; d < dim; d++)
			centers[c + 1]->x[d] = vectors[selected_idx]->x[d];
	}

	/* Step 2: Lloyd's iterations */
	for (int iter = 0; iter < max_iter; iter++)
	{
		CHECK_FOR_INTERRUPTS();

		/* Assign points */
		for (int i = 0; i < n; i++)
		{
			float		min_dist = FLT_MAX;
			int			best_c = 0;

			for (int c = 0; c < k; c++)
			{
				float		d = VectorDistSq(dim, vectors[i]->x, centers[c]->x);

				if (d < min_dist)
				{
					min_dist = d;
					best_c = c;
				}
			}
			assignments[i] = best_c;
		}

		/* Compute new centroids */
		memset(cluster_counts, 0, (Size) k * sizeof(int));
		memset(cluster_sums, 0, (Size) k * (Size) dim * sizeof(float));

		for (int i = 0; i < n; i++)
		{
			int			c = assignments[i];

			cluster_counts[c]++;
			for (int d = 0; d < dim; d++)
				cluster_sums[c * dim + d] += vectors[i]->x[d];
		}

		movement = 0.0;
		for (int c = 0; c < k; c++)
		{
			if (cluster_counts[c] > 0)
			{
				float		shift = 0.0f;

				for (int d = 0; d < dim; d++)
				{
					float		new_val = cluster_sums[c * dim + d] / (float) cluster_counts[c];
					float		diff = new_val - centers[c]->x[d];

					shift += diff * diff;
					centers[c]->x[d] = new_val;
				}
				if (shift > movement)
					movement = shift;
			}
		}

		if (movement < 1e-6)
			break;
	}

	pfree(weights);
	pfree(cluster_counts);
	pfree(cluster_sums);
}

/*
 * Structure for SRF k-Means
 */
typedef struct KMeansState
{
	int			k;
	int			current;
	Vector	  **centers;
} KMeansState;

/*
 * vector_kmeans(vector[], integer, integer DEFAULT 20)
 * RETURNS TABLE(cluster_id integer, centroid vector)
 */
PG_FUNCTION_INFO_V1(vector_kmeans);
Datum
vector_kmeans(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	KMeansState *state;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;
		ArrayType  *array = PG_GETARG_ARRAYTYPE_P(0);
		int			k = PG_GETARG_INT32(1);
		int			max_iter = PG_NARGS() > 2 ? PG_GETARG_INT32(2) : 20;
		int			num_vectors;
		int			dim;
		Vector	  **vectors;
		int		   *assignments;

		if (k <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("k must be greater than 0")));

		if (max_iter <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("max_iter must be greater than 0")));

		vectors = ExtractVectorArray(array, &num_vectors, &dim);
		if (num_vectors == 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("vector array cannot be empty")));

		if (k > num_vectors)
			k = num_vectors;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("return type must be a row type")));

		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		state = palloc_array_checked(KMeansState, 1);
		state->k = k;
		state->current = 0;
		state->centers = palloc_array_checked(Vector *, (Size) k);
		assignments = palloc_array_checked(int, (Size) num_vectors);

		RunKMeansInternal(vectors, num_vectors, dim, k, max_iter, state->centers, assignments);

		pfree(assignments);
		funcctx->user_fctx = state;
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	state = (KMeansState *) funcctx->user_fctx;

	if (state->current < state->k)
	{
		Datum		values[2];
		bool		nulls[2] = {false, false};
		HeapTuple	tuple;
		Datum		result;

		values[0] = Int32GetDatum(state->current);
		values[1] = PointerGetDatum(state->centers[state->current]);

		tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
		result = HeapTupleGetDatum(tuple);

		state->current++;
		SRF_RETURN_NEXT(funcctx, result);
	}
	else
	{
		SRF_RETURN_DONE(funcctx);
	}
}

/*
 * vector_kmeans_assign(vector[], vector[])
 * RETURNS integer[]
 */
PG_FUNCTION_INFO_V1(vector_kmeans_assign);
Datum
vector_kmeans_assign(PG_FUNCTION_ARGS)
{
	ArrayType  *vec_array = PG_GETARG_ARRAYTYPE_P(0);
	ArrayType  *cent_array = PG_GETARG_ARRAYTYPE_P(1);
	int			n_vectors;
	int			dim_v;
	int			n_centroids;
	int			dim_c;
	Vector	  **vectors;
	Vector	  **centroids;
	Datum	   *result_datums;
	ArrayType  *result;

	vectors = ExtractVectorArray(vec_array, &n_vectors, &dim_v);
	centroids = ExtractVectorArray(cent_array, &n_centroids, &dim_c);

	if (dim_v != dim_c)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions do not match: %d and %d", dim_v, dim_c)));

	if (n_centroids == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("centroids array cannot be empty")));

	result_datums = palloc_array_checked(Datum, (Size) n_vectors);

	for (int i = 0; i < n_vectors; i++)
	{
		float		min_dist = FLT_MAX;
		int			best_c = 0;

		for (int c = 0; c < n_centroids; c++)
		{
			float		d = VectorDistSq(dim_v, vectors[i]->x, centroids[c]->x);

			if (d < min_dist)
			{
				min_dist = d;
				best_c = c;
			}
		}
		result_datums[i] = Int32GetDatum(best_c);
	}

	result = construct_array(result_datums, n_vectors, INT4OID, sizeof(int32), true, TYPALIGN_INT);
	pfree(result_datums);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * DBSCAN State for SRF
 */
typedef struct DBSCANState
{
	int			num_points;
	int			current;
	int		   *labels;
} DBSCANState;

/*
 * vector_dbscan(vector[], float8 eps, integer min_samples)
 * RETURNS TABLE(item_index integer, cluster_id integer)
 */
PG_FUNCTION_INFO_V1(vector_dbscan);
Datum
vector_dbscan(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	DBSCANState *state;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;
		ArrayType  *array = PG_GETARG_ARRAYTYPE_P(0);
		float8		eps = PG_GETARG_FLOAT8(1);
		int			min_samples = PG_GETARG_INT32(2);
		int			n;
		int			dim;
		Vector	  **vectors;
		float		eps_sq = (float) (eps * eps);
		int		   *labels;
		int			cluster_id = 1;
		int		   *queue;
		int			queue_cap;

		if (eps <= 0.0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("eps must be greater than 0")));

		if (min_samples <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("min_samples must be greater than 0")));

		vectors = ExtractVectorArray(array, &n, &dim);
		if (n == 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("vector array cannot be empty")));

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("return type must be a row type")));

		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		labels = palloc_array_checked(int, (Size) n);
		memset(labels, 0, (Size) n * sizeof(int));

		queue_cap = n;
		queue = palloc_array_checked(int, (Size) queue_cap);

		for (int i = 0; i < n; i++)
		{
			int			n_neighbors = 0;

			CHECK_FOR_INTERRUPTS();

			if (labels[i] != 0)
				continue;

			/* Find neighbors */
			for (int j = 0; j < n; j++)
			{
				if (VectorDistSq(dim, vectors[i]->x, vectors[j]->x) <= eps_sq)
					queue[n_neighbors++] = j;
			}

			if (n_neighbors < min_samples)
			{
				labels[i] = -1;	/* noise */
			}
			else
			{
				int			head = 0;

				labels[i] = cluster_id;

				while (head < n_neighbors)
				{
					int			p = queue[head++];

					if (labels[p] == -1)
						labels[p] = cluster_id;

					if (labels[p] != 0)
						continue;

					labels[p] = cluster_id;

					/* Find p's neighbors */
					{
						int			p_neighbors = 0;

						for (int j = 0; j < n; j++)
						{
							if (VectorDistSq(dim, vectors[p]->x, vectors[j]->x) <= eps_sq)
								p_neighbors++;
						}

						if (p_neighbors >= min_samples)
						{
							for (int j = 0; j < n; j++)
							{
								if (VectorDistSq(dim, vectors[p]->x, vectors[j]->x) <= eps_sq)
								{
									/* Add to queue if not present */
									bool		found = false;

									for (int q = 0; q < n_neighbors; q++)
									{
										if (queue[q] == j)
										{
											found = true;
											break;
										}
									}
									if (!found && n_neighbors < queue_cap)
										queue[n_neighbors++] = j;
								}
							}
						}
					}
				}
				cluster_id++;
			}
		}

		pfree(queue);

		state = palloc_array_checked(DBSCANState, 1);
		state->num_points = n;
		state->current = 0;
		state->labels = labels;

		funcctx->user_fctx = state;
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	state = (DBSCANState *) funcctx->user_fctx;

	if (state->current < state->num_points)
	{
		Datum		values[2];
		bool		nulls[2] = {false, false};
		HeapTuple	tuple;
		Datum		result;

		values[0] = Int32GetDatum(state->current + 1);	/* 1-based index */
		values[1] = Int32GetDatum(state->labels[state->current]);

		tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
		result = HeapTupleGetDatum(tuple);

		state->current++;
		SRF_RETURN_NEXT(funcctx, result);
	}
	else
	{
		SRF_RETURN_DONE(funcctx);
	}
}

/*
 * SPANN Assignment:
 * spann_assign(vector, vector[], float8 alpha DEFAULT 0.15, integer max_centroids DEFAULT 4)
 * RETURNS integer[] (1-based centroid indices)
 */
typedef struct CentroidDist
{
	int			index;
	float		dist;
} CentroidDist;

static int
CompareCentroidDist(const void *a, const void *b)
{
	float		diff = ((const CentroidDist *) a)->dist - ((const CentroidDist *) b)->dist;

	return (diff > 0.0f) - (diff < 0.0f);
}

PG_FUNCTION_INFO_V1(spann_assign);
Datum
spann_assign(PG_FUNCTION_ARGS)
{
	Vector	   *vec = PG_GETARG_VECTOR_P(0);
	ArrayType  *cent_array = PG_GETARG_ARRAYTYPE_P(1);
	float8		alpha = PG_NARGS() > 2 ? PG_GETARG_FLOAT8(2) : 0.15;
	int			max_centroids = PG_NARGS() > 3 ? PG_GETARG_INT32(3) : 4;
	int			n_centroids;
	int			dim_c;
	Vector	  **centroids;
	CentroidDist *dists;
	float		min_dist;
	float		dist_threshold;
	int			assigned_count = 0;
	Datum	   *result_datums;
	ArrayType  *result;

	if (alpha < 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("alpha must be non-negative")));

	if (max_centroids <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("max_centroids must be greater than 0")));

	centroids = ExtractVectorArray(cent_array, &n_centroids, &dim_c);
	if (n_centroids == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("centroids array cannot be empty")));

	if (vec->dim != dim_c)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions do not match: %d and %d", vec->dim, dim_c)));

	dists = palloc_array_checked(CentroidDist, (Size) n_centroids);
	for (int i = 0; i < n_centroids; i++)
	{
		dists[i].index = i + 1;	/* 1-based index */
		dists[i].dist = VectorDistL2(vec->dim, vec->x, centroids[i]->x);
	}

	qsort(dists, n_centroids, sizeof(CentroidDist), CompareCentroidDist);

	min_dist = dists[0].dist;
	dist_threshold = (float) ((1.0 + alpha) * min_dist);

	result_datums = palloc_array_checked(Datum, (Size) max_centroids);
	for (int i = 0; i < n_centroids && assigned_count < max_centroids; i++)
	{
		if (dists[i].dist <= dist_threshold || assigned_count == 0)
		{
			result_datums[assigned_count++] = Int32GetDatum(dists[i].index);
		}
		else
		{
			break;
		}
	}

	result = construct_array(result_datums, assigned_count, INT4OID, sizeof(int32), true, TYPALIGN_INT);
	pfree(dists);
	pfree(result_datums);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * SPANN Query / Candidate Re-ranking:
 * spann_query(vector query_vec, vector[] candidate_vecs, integer top_k DEFAULT 10)
 * RETURNS integer[] (top-k 1-based indices in candidate array sorted by distance)
 */
PG_FUNCTION_INFO_V1(spann_query);
Datum
spann_query(PG_FUNCTION_ARGS)
{
	Vector	   *query = PG_GETARG_VECTOR_P(0);
	ArrayType  *cand_array = PG_GETARG_ARRAYTYPE_P(1);
	int			top_k = PG_NARGS() > 2 ? PG_GETARG_INT32(2) : 10;
	int			n_cands;
	int			dim_c;
	Vector	  **candidates;
	CentroidDist *dists;
	int			return_count;
	Datum	   *result_datums;
	ArrayType  *result;

	if (top_k <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("top_k must be greater than 0")));

	candidates = ExtractVectorArray(cand_array, &n_cands, &dim_c);
	if (n_cands == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("candidates array cannot be empty")));

	if (query->dim != dim_c)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector dimensions do not match: %d and %d", query->dim, dim_c)));

	dists = palloc_array_checked(CentroidDist, (Size) n_cands);
	for (int i = 0; i < n_cands; i++)
	{
		dists[i].index = i + 1;
		dists[i].dist = VectorDistSq(query->dim, query->x, candidates[i]->x);
	}

	qsort(dists, n_cands, sizeof(CentroidDist), CompareCentroidDist);

	return_count = Min(top_k, n_cands);
	result_datums = palloc_array_checked(Datum, (Size) return_count);
	for (int i = 0; i < return_count; i++)
		result_datums[i] = Int32GetDatum(dists[i].index);

	result = construct_array(result_datums, return_count, INT4OID, sizeof(int32), true, TYPALIGN_INT);
	pfree(dists);
	pfree(result_datums);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * Jacobi Eigenvalue Algorithm for Real Symmetric Matrix
 * Finds all eigenvalues and eigenvectors of A (n x n)
 */
static void
JacobiEigen(double *A, int n, double *eigenvalues, double *V)
{
	int			max_rotations = 100 * n * n;
	int			rotations = 0;

	/* Initialize V as identity matrix */
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			V[i * n + j] = (i == j) ? 1.0 : 0.0;
	}

	while (rotations < max_rotations)
	{
		/* Find largest off-diagonal element */
		double		max_offdiag = 0.0;
		int			p = 0,
					q = 1;

		for (int i = 0; i < n; i++)
		{
			for (int j = i + 1; j < n; j++)
			{
				double		val = fabs(A[i * n + j]);

				if (val > max_offdiag)
				{
					max_offdiag = val;
					p = i;
					q = j;
				}
			}
		}

		if (max_offdiag < 1e-12)
			break;

		/* Compute rotation */
		{
			double		app = A[p * n + p];
			double		aqq = A[q * n + q];
			double		apq = A[p * n + q];
			double		theta = (aqq - app) / (2.0 * apq);
			double		t;
			double		c,
						s;

			if (theta >= 0.0)
				t = 1.0 / (theta + sqrt(1.0 + theta * theta));
			else
				t = -1.0 / (-theta + sqrt(1.0 + theta * theta));

			c = 1.0 / sqrt(1.0 + t * t);
			s = t * c;

			/* Update diagonal and zero out (p, q) */
			A[p * n + p] = app - t * apq;
			A[q * n + q] = aqq + t * apq;
			A[p * n + q] = 0.0;
			A[q * n + p] = 0.0;

			/* Update rest of matrix A */
			for (int r = 0; r < n; r++)
			{
				if (r != p && r != q)
				{
					double		arp = A[r * n + p];
					double		arq = A[r * n + q];

					A[r * n + p] = c * arp - s * arq;
					A[p * n + r] = A[r * n + p];
					A[r * n + q] = s * arp + c * arq;
					A[q * n + r] = A[r * n + q];
				}
			}

			/* Update eigenvector matrix V */
			for (int r = 0; r < n; r++)
			{
				double		vrp = V[r * n + p];
				double		vrq = V[r * n + q];

				V[r * n + p] = c * vrp - s * vrq;
				V[r * n + q] = s * vrp + c * vrq;
			}
		}

		rotations++;
	}

	for (int i = 0; i < n; i++)
		eigenvalues[i] = A[i * n + i];
}

/*
 * Spectral Clustering State for SRF
 */
typedef struct SpectralState
{
	int			num_points;
	int			current;
	int		   *labels;
	Vector	  **projections;
} SpectralState;

/*
 * vector_spectral_clustering(vector[], integer k_clusters, integer n_neighbors DEFAULT 5, integer n_components DEFAULT 2)
 * RETURNS TABLE(item_index integer, cluster_id integer, spectral_embedding vector)
 */
PG_FUNCTION_INFO_V1(vector_spectral_clustering);
Datum
vector_spectral_clustering(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	SpectralState *state;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		TupleDesc	tupdesc;
		ArrayType  *array = PG_GETARG_ARRAYTYPE_P(0);
		int			k_clusters = PG_GETARG_INT32(1);
		int			n_neighbors = PG_NARGS() > 2 ? PG_GETARG_INT32(2) : 5;
		int			n_components = PG_NARGS() > 3 ? PG_GETARG_INT32(3) : 2;
		int			n;
		int			dim;
		Vector	  **vectors;
		double	   *W;
		double	   *D_inv_sqrt;
		double	   *L_sym;
		double	   *eigenvalues;
		double	   *V;
		int		   *order;
		Vector	  **proj_vectors;
		Vector	  **spectral_centers;
		int		   *labels;

		if (k_clusters <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("k_clusters must be greater than 0")));

		if (n_neighbors <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("n_neighbors must be greater than 0")));

		if (n_components <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("n_components must be greater than 0")));

		vectors = ExtractVectorArray(array, &n, &dim);
		if (n == 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("vector array cannot be empty")));

		if (k_clusters > n)
			k_clusters = n;
		if (n_components > n)
			n_components = n;
		if (n_neighbors >= n)
			n_neighbors = n - 1;
		if (n_neighbors < 1)
			n_neighbors = 1;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("return type must be a row type")));

		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		/* 1. Build Affinity Matrix W using k-NN */
		W = palloc0_array(double, (Size) n * (Size) n);
		{
			CentroidDist *dists = palloc_array_checked(CentroidDist, (Size) n);
			double		sum_sigma = 0.0;
			int			sigma_count = 0;
			double		sigma_sq;

			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < n; j++)
				{
					dists[j].index = j;
					dists[j].dist = VectorDistSq(dim, vectors[i]->x, vectors[j]->x);
				}
				qsort(dists, n, sizeof(CentroidDist), CompareCentroidDist);

				for (int k = 1; k <= n_neighbors && k < n; k++)
				{
					sum_sigma += dists[k].dist;
					sigma_count++;
				}
			}

			sigma_sq = (sigma_count > 0 && sum_sigma > 0.0) ? (sum_sigma / sigma_count) : 1.0;

			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < n; j++)
				{
					dists[j].index = j;
					dists[j].dist = VectorDistSq(dim, vectors[i]->x, vectors[j]->x);
				}
				qsort(dists, n, sizeof(CentroidDist), CompareCentroidDist);

				for (int k = 1; k <= n_neighbors && k < n; k++)
				{
					int			j = dists[k].index;
					double		aff = exp(-(double) dists[k].dist / (2.0 * sigma_sq));

					if (aff > W[i * n + j])
						W[i * n + j] = aff;
				}
			}
			pfree(dists);

			/* Symmetrize W */
			for (int i = 0; i < n; i++)
			{
				for (int j = i + 1; j < n; j++)
				{
					double		avg_aff = 0.5 * (W[i * n + j] + W[j * n + i]);

					W[i * n + j] = avg_aff;
					W[j * n + i] = avg_aff;
				}
			}
		}

		/* 2. Compute Degrees and Normalized Laplacian L_sym = I - D^(-1/2) W D^(-1/2) */
		D_inv_sqrt = palloc_array_checked(double, (Size) n);
		for (int i = 0; i < n; i++)
		{
			double		d = 0.0;

			for (int j = 0; j < n; j++)
				d += W[i * n + j];

			D_inv_sqrt[i] = (d > 1e-10) ? (1.0 / sqrt(d)) : 0.0;
		}

		L_sym = palloc_array_checked(double, (Size) n * (Size) n);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				double		l = -D_inv_sqrt[i] * W[i * n + j] * D_inv_sqrt[j];

				if (i == j)
					l += 1.0;
				L_sym[i * n + j] = l;
			}
		}

		/* 3. Compute eigenvalues and eigenvectors */
		eigenvalues = palloc_array_checked(double, (Size) n);
		V = palloc_array_checked(double, (Size) n * (Size) n);
		JacobiEigen(L_sym, n, eigenvalues, V);

		/* Sort eigenvectors by ascending eigenvalues */
		order = palloc_array_checked(int, (Size) n);
		for (int i = 0; i < n; i++)
			order[i] = i;

		for (int i = 0; i < n - 1; i++)
		{
			for (int j = i + 1; j < n; j++)
			{
				if (eigenvalues[order[i]] > eigenvalues[order[j]])
				{
					int			tmp = order[i];

					order[i] = order[j];
					order[j] = tmp;
				}
			}
		}

		/* 4. Construct row-normalized spectral projection vectors Y */
		proj_vectors = palloc_array_checked(Vector *, (Size) n);
		for (int i = 0; i < n; i++)
		{
			double		row_norm_sq = 0.0;

			proj_vectors[i] = InitVector(n_components);
			for (int d = 0; d < n_components; d++)
			{
				int			col = order[d];
				double		val = V[i * n + col];

				proj_vectors[i]->x[d] = (float) val;
				row_norm_sq += val * val;
			}

			if (row_norm_sq > 1e-12)
			{
				float		inv_norm = 1.0f / sqrtf((float) row_norm_sq);

				for (int d = 0; d < n_components; d++)
					proj_vectors[i]->x[d] *= inv_norm;
			}
		}

		/* 5. Cluster projected vectors with k-means */
		spectral_centers = palloc_array_checked(Vector *, (Size) k_clusters);
		labels = palloc_array_checked(int, (Size) n);
		RunKMeansInternal(proj_vectors, n, n_components, k_clusters, 25, spectral_centers, labels);

		/* Cleanup intermediate buffers */
		pfree(W);
		pfree(D_inv_sqrt);
		pfree(L_sym);
		pfree(eigenvalues);
		pfree(V);
		pfree(order);
		pfree(spectral_centers);

		state = palloc_array_checked(SpectralState, 1);
		state->num_points = n;
		state->current = 0;
		state->labels = labels;
		state->projections = proj_vectors;

		funcctx->user_fctx = state;
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	state = (SpectralState *) funcctx->user_fctx;

	if (state->current < state->num_points)
	{
		Datum		values[3];
		bool		nulls[3] = {false, false, false};
		HeapTuple	tuple;
		Datum		result;

		values[0] = Int32GetDatum(state->current + 1);	/* 1-based index */
		values[1] = Int32GetDatum(state->labels[state->current]);
		values[2] = PointerGetDatum(state->projections[state->current]);

		tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
		result = HeapTupleGetDatum(tuple);

		state->current++;
		SRF_RETURN_NEXT(funcctx, result);
	}
	else
	{
		SRF_RETURN_DONE(funcctx);
	}
}

/*
 * Out-of-sample projection into spectral space:
 * vector_spectral_project(vector query_vec, vector[] sample_vecs, vector[] sample_projections, integer k_neighbors DEFAULT 5)
 * RETURNS vector
 */
PG_FUNCTION_INFO_V1(vector_spectral_project);
Datum
vector_spectral_project(PG_FUNCTION_ARGS)
{
	Vector	   *query = PG_GETARG_VECTOR_P(0);
	ArrayType  *samples_array = PG_GETARG_ARRAYTYPE_P(1);
	ArrayType  *projs_array = PG_GETARG_ARRAYTYPE_P(2);
	int			k_neighbors = PG_NARGS() > 3 ? PG_GETARG_INT32(3) : 5;
	int			n_samples;
	int			dim_s;
	int			n_projs;
	int			dim_p;
	Vector	  **sample_vecs;
	Vector	  **sample_projs;
	CentroidDist *dists;
	double		sum_dist = 0.0;
	double		sigma_sq;
	double		total_weight = 0.0;
	float	   *accum;
	float		norm_sq = 0.0f;
	Vector	   *result;

	if (k_neighbors <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("k_neighbors must be greater than 0")));

	sample_vecs = ExtractVectorArray(samples_array, &n_samples, &dim_s);
	sample_projs = ExtractVectorArray(projs_array, &n_projs, &dim_p);

	if (n_samples == 0 || n_projs == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("samples and projections arrays cannot be empty")));

	if (n_samples != n_projs)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("samples and projections arrays must have equal length (%d vs %d)",
						n_samples, n_projs)));

	if (query->dim != dim_s)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("query vector dimension (%d) does not match sample vector dimension (%d)",
						query->dim, dim_s)));

	if (k_neighbors > n_samples)
		k_neighbors = n_samples;

	dists = palloc_array_checked(CentroidDist, (Size) n_samples);
	for (int i = 0; i < n_samples; i++)
	{
		dists[i].index = i;
		dists[i].dist = VectorDistSq(query->dim, query->x, sample_vecs[i]->x);
	}

	qsort(dists, n_samples, sizeof(CentroidDist), CompareCentroidDist);

	for (int k = 0; k < k_neighbors; k++)
		sum_dist += dists[k].dist;

	sigma_sq = (sum_dist > 0.0) ? (sum_dist / (double) k_neighbors) : 1.0;

	accum = palloc0_array(float, (Size) dim_p);
	for (int k = 0; k < k_neighbors; k++)
	{
		int			idx = dists[k].index;
		double		w = exp(-(double) dists[k].dist / (2.0 * sigma_sq));

		total_weight += w;
		for (int d = 0; d < dim_p; d++)
			accum[d] += (float) (w * sample_projs[idx]->x[d]);
	}

	result = InitVector(dim_p);
	for (int d = 0; d < dim_p; d++)
	{
		float		val = (total_weight > 0.0) ? (accum[d] / (float) total_weight) : accum[d];

		result->x[d] = val;
		norm_sq += val * val;
	}

	if (norm_sq > 1e-12)
	{
		float		inv_norm = 1.0f / sqrtf(norm_sq);

		for (int d = 0; d < dim_p; d++)
			result->x[d] *= inv_norm;
	}

	pfree(dists);
	pfree(accum);

	PG_RETURN_POINTER(result);
}
