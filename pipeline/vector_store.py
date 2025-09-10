from pymilvus import MilvusClient, DataType
from pymilvus.model.sparse import BM25EmbeddingFunction
from config import Config
from sparse_utils import get_sparse_row
import hybrid_rerank
import time

class VectorStore:
    def __init__(self, uri=Config.MILVUS_URI):
        self.client = MilvusClient(uri=uri)
        self.collection_name = Config.COLLECTION_NAME

    def ensure_collection(self, dim=Config.DENSE_DIM):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2000, enable_analyzer=True)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        self.client.create_collection(collection_name=self.collection_name, schema=schema)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="IVF_FLAT",
            metric_type="L2"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )

    def add_documents(self, text_chunks, progress_callback=None):
        start_time = time.time()
        corpus = [chunk["text"] for chunk in text_chunks]
        if not corpus or all(not t.strip() for t in corpus):
            raise ValueError("Corpus is empty. Please upload non-empty documents.")

        sparse_embedder = BM25EmbeddingFunction(corpus=corpus)
        t1 = time.time()
        dense_vectors = hybrid_rerank.embed_dense_batch(corpus)
        t2 = time.time()
        sparse_vectors = sparse_embedder.encode_documents(corpus)
        t3 = time.time()

        total = len(text_chunks)
        data = []
        for i, chunk in enumerate(text_chunks):
            sparse_row = get_sparse_row(sparse_vectors, i)
            if not sparse_row:
                sparse_row = {0: 0.0}
            sparse_row = {int(k): float(v) for k, v in sparse_row.items()}
            assert all(isinstance(k, int) and isinstance(v, float) for k, v in sparse_row.items())
            data.append({
                "text": chunk["text"],
                "dense_vector": dense_vectors[i],
                "sparse_vector": sparse_row,
                "metadata": chunk["metadata"]
            })
            # Smoother progress: update every ~2% instead of 1%
            if progress_callback and ((i + 1) % max(1, total // 50) == 0 or (i + 1) == total):
                percent = int((i + 1) / total * 100)
                progress_callback(percent)
        t4 = time.time()
        self.client.insert(collection_name=self.collection_name, data=data)
        t5 = time.time()
        self.client.load_collection(self.collection_name)
        t6 = time.time()
        print(f"Embedding dense batch time: {t2 - t1:.2f}s")
        print(f"Encoding sparse vectors time: {t3 - t2:.2f}s")
        print(f"Data preparation time: {t4 - t3:.2f}s")
        print(f"Milvus insert time: {t5 - t4:.2f}s")
        print(f"Milvus load collection time: {t6 - t5:.2f}s")
        print(f"Total add_documents time: {t6 - start_time:.2f}s")
        return corpus

    def get_corpus(self):
        batch_size = 16384
        all_texts = []
        offset = 0
        while True:
            results = self.client.query(
                self.collection_name,
                output_fields=["text"],
                offset=offset,
                limit=batch_size
            )
            if not results:
                break
            all_texts.extend([item["text"] for item in results])
            offset += len(results)
            if len(results) < batch_size:
                break
        return all_texts
