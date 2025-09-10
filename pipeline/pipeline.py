from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# Load embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Setup Chroma vector store - specify persist_directory to store locally
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# Load FinGPT LLM (example assumes HuggingFace pipeline integration)
hf_pipeline = pipeline(
    "text-generation",
    model="D:\\FinGPT\\model\\directory",
    max_length=512,
    do_sample=True,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Build retrieval augmented QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

# Example query
query = "Explain the risk factors for borrower ID 12345"
result = qa_chain.run(query)
print(result)