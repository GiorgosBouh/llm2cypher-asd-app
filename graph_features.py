# graph_features.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from neo4j import GraphDatabase

class GraphFeatureExtractor:
    def __init__(self, neo4j_uri, user, password):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def compute_similarity_to_asd(self, new_embedding, asd_embeddings):
        similarities = cosine_similarity([new_embedding], asd_embeddings)[0]
        return np.mean(similarities)

    def get_asd_embeddings(self):
        query = """
        MATCH (c:Case)-[:SCREENED_FOR]->(:ASD_Trait {value: 'Yes'})
        WHERE c.embedding IS NOT NULL
        RETURN c.embedding AS embedding
        """
        with self.driver.session() as session:
            result = session.run(query)
            return np.array([r["embedding"] for r in result])

    def count_shared_demographics_with_asd(self, case_no):
        query = """
        MATCH (c1:Case {Case_No: $case_no})-[:HAS_DEMOGRAPHIC]->(d:DemographicAttribute)<-[:HAS_DEMOGRAPHIC]-(c2:Case)
        WHERE (c2)-[:SCREENED_FOR]->(:ASD_Trait {value: 'Yes'}) AND c1 <> c2
        RETURN COUNT(DISTINCT c2) AS count
        """
        with self.driver.session() as session:
            result = session.run(query, case_no=case_no).single()
            return result["count"] if result else 0

    def cluster_embeddings(self, embeddings, n_clusters=10):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(embeddings)

# Example usage:
# gfe = GraphFeatureExtractor("bolt://localhost:7687", "neo4j", "password")
# asd_embs = gfe.get_asd_embeddings()
# sim = gfe.compute_similarity_to_asd(new_emb, asd_embs)
# shared = gfe.count_shared_demographics_with_asd(1852)
# cluster_id = gfe.cluster_embeddings(embeddings)[index_of_case]
# gfe.close()
