from pymilvus import AnnSearchRequest, RRFRanker
from pymilvus.model.sparse import BM25EmbeddingFunction
from embeddings import embed_dense, embed_dense_batch
from sparse_utils import get_sparse_row

def hybrid_search(query, client, collection_name, corpus, limit=5):
    if not corpus:
        raise ValueError("BM25 corpus is empty! Did you build the knowledge base?")
    sparse_embedder = BM25EmbeddingFunction(corpus=corpus)
    dense_vec = embed_dense(query)
    query_sparse_matrix = sparse_embedder.encode_documents([query])
    sparse_vec = get_sparse_row(query_sparse_matrix, 0)
    if not sparse_vec:
        sparse_vec = {0: 0.0}
    sparse_vec = {int(k): float(v) for k, v in sparse_vec.items()}
    assert all(isinstance(k, int) and isinstance(v, float) for k, v in sparse_vec.items())

    dense_req = AnnSearchRequest(
        data=[dense_vec],
        anns_field="dense_vector",
        param={"metric_type": "L2"},
        limit=10
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vec],
        anns_field="sparse_vector",
        param={"metric_type": "IP"},
        limit=10
    )

    results = client.hybrid_search(
        collection_name,
        [dense_req, sparse_req],
        RRFRanker(),
        limit=limit,
        output_fields=["text", "metadata"]
    )

    return results



