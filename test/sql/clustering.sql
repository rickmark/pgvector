-- k-means clustering
SELECT * FROM vector_kmeans(ARRAY['[1,2]'::vector, '[1,2.1]', '[10,10]', '[10,10.2]'], 2);
SELECT vector_kmeans_assign(ARRAY['[1,2]'::vector, '[10,10]', '[1,2.5]'], ARRAY['[1,2]'::vector, '[10,10]']);

-- k-means edge cases
SELECT * FROM vector_kmeans(ARRAY['[1,2,3]'::vector], 1);
SELECT * FROM vector_kmeans(ARRAY['[1,2]'::vector, '[3,4]'], 5);
SELECT * FROM vector_kmeans(ARRAY[]::vector[], 2);
SELECT * FROM vector_kmeans(ARRAY['[1,2]'::vector, NULL], 2);
SELECT * FROM vector_kmeans(ARRAY['[1,2]'::vector, '[1,2,3]'], 2);
SELECT * FROM vector_kmeans(ARRAY['[1,2]'::vector], 0);
SELECT * FROM vector_kmeans(ARRAY['[1,2]'::vector], 1, 0);

-- DBSCAN clustering
SELECT * FROM vector_dbscan(ARRAY['[1,2]'::vector, '[1,2.1]', '[10,10]', '[10,10.2]', '[100,100]'], 1.0, 2);
SELECT * FROM vector_dbscan(ARRAY['[0,0]'::vector, '[0,0.1]', '[0,0.2]', '[5,5]', '[5,5.1]'], 0.5, 2);

-- DBSCAN edge cases
SELECT * FROM vector_dbscan(ARRAY['[1,2]'::vector], 0.5, 1);
SELECT * FROM vector_dbscan(ARRAY['[1,2]'::vector], 0.0, 1);
SELECT * FROM vector_dbscan(ARRAY['[1,2]'::vector], 0.5, 0);
SELECT * FROM vector_dbscan(ARRAY[]::vector[], 0.5, 2);

-- SPANN assignment & candidate query
SELECT spann_assign('[1,2]'::vector, ARRAY['[1,2]'::vector, '[1.1,2.1]', '[10,10]'], 0.2, 3);
SELECT spann_assign('[1,2]'::vector, ARRAY['[1,2]'::vector, '[1.1,2.1]', '[10,10]'], 0.01, 3);
SELECT spann_query('[1,2]'::vector, ARRAY['[10,10]'::vector, '[1,2.1]', '[1,2]'], 2);
SELECT spann_query('[1,2]'::vector, ARRAY['[10,10]'::vector, '[1,2.1]', '[1,2]'], 10);

-- SPANN edge cases
SELECT spann_assign('[1,2]'::vector, ARRAY['[1,2,3]'::vector], 0.15, 2);
SELECT spann_assign('[1,2]'::vector, ARRAY['[1,2]'::vector], -0.1, 2);
SELECT spann_assign('[1,2]'::vector, ARRAY['[1,2]'::vector], 0.15, 0);
SELECT spann_assign('[1,2]'::vector, ARRAY[]::vector[], 0.15, 2);

-- Spectral clustering
SELECT item_index, cluster_id, vector_dims(spectral_embedding) AS proj_dims
FROM vector_spectral_clustering(ARRAY['[1,2]'::vector, '[1,2.1]', '[10,10]', '[10,10.2]'], 2, 2, 2);

-- Spectral project (out of sample)
SELECT vector_spectral_project(
    '[1,2.05]'::vector,
    ARRAY['[1,2]'::vector, '[1,2.1]', '[10,10]', '[10,10.2]'],
    ARRAY['[1,0]'::vector, '[1,0]', '[0,1]', '[0,1]'],
    2
);

-- Spectral edge cases
SELECT * FROM vector_spectral_clustering(ARRAY['[1,2]'::vector], 0, 1, 1);
SELECT * FROM vector_spectral_clustering(ARRAY['[1,2]'::vector], 1, 0, 1);
SELECT * FROM vector_spectral_clustering(ARRAY['[1,2]'::vector], 1, 1, 0);
SELECT vector_spectral_project('[1,2]'::vector, ARRAY['[1,2]'::vector], ARRAY['[1,0]'::vector, '[0,1]'], 1);
SELECT vector_spectral_project('[1,2,3]'::vector, ARRAY['[1,2]'::vector], ARRAY['[1,0]'::vector], 1);
