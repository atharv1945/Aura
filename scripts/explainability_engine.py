import pandas as pd
import os
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = "data/raw/"
PROCESSED_DIR = "data/processed"
COMMUNICATIONS_PATH = os.path.join(DATA_DIR, "communications.csv")
EVENTS_PATH = os.path.join(DATA_DIR, "events.csv")
TRANSACTIONS_PATH = os.path.join(DATA_DIR, "transactions.csv")
BORROWERS_ENHANCED_PATH = os.path.join(PROCESSED_DIR, "borrowers_enhanced.csv")
AURA_SCORES_PATH = os.path.join(PROCESSED_DIR, "aura_risk_scores.csv")

# Load environment variables (for GOOGLE_API_KEY)
load_dotenv()

# --- PHASE 1: PRE-RETRIEVAL (Query Augmentation) ---

def augment_query(borrower_id: str, events_df: pd.DataFrame, borrowers_df: pd.DataFrame) -> str:
    """Augments a simple query with specific context about the borrower."""
    print(f"\n--- Phase 1: Augmenting Query for {borrower_id} ---")
    
    borrower_events = events_df[events_df['borrower_id'] == borrower_id].copy()
    if borrower_events.empty:
        print("No events found for this borrower.")
        return f"Analyze the overall credit risk for {borrower_id} based on their profile."

    borrower_events['severity_score'] = borrower_events['severity'].map({'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4})
    top_events = borrower_events.nlargest(2, 'severity_score')
    event_str = " and ".join(top_events['event_type'].tolist())
    
    archetype = borrowers_df[borrowers_df['borrower_id'] == borrower_id]['behavioral_archetype'].iloc[0]
    
    augmented_query = f"Analyze the credit risk for {borrower_id}, focusing on recent critical events like '{event_str}' and considering their behavioral profile as a '{archetype}'."
    print(f"Augmented Query: \"{augmented_query}\"")
    return augmented_query

# --- PHASE 2: HYBRID RETRIEVAL ---

class HybridRetriever:
    """Retrieves evidence using both dense (semantic) and sparse (keyword) search."""
    def __init__(self, comms_df, events_df, transactions_df):
        print("\n--- Phase 2a: Initializing Hybrid Retriever ---")
        self.events_df = events_df
        self.transactions_df = transactions_df
        self.comms_df = comms_df
        
        print("Loading sentence transformer model for dense retrieval...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Building FAISS index for communications data...")
        comms_embeddings = self.model.encode(self.comms_df['comm_text'].tolist(), convert_to_tensor=True)
        self.index = faiss.IndexFlatL2(comms_embeddings.shape[1])
        self.index.add(comms_embeddings.cpu().detach().numpy())
        print("Retriever initialized.")

    def _dense_retrieval(self, query: str, k: int = 5):
        query_embedding = self.model.encode([query])
        _, indices = self.index.search(query_embedding, k)
        return self.comms_df.iloc[indices[0]].to_dict('records')

    def _sparse_retrieval(self, borrower_id: str):
        # Simplified keyword search on structured data
        top_events = self.events_df[self.events_df['borrower_id'] == borrower_id].nlargest(3, 'severity_score', keep='all')
        
        # Find top 5 largest debit transactions
        borrower_txs = self.transactions_df[self.transactions_df['borrower_id'] == borrower_id]
        top_transactions = borrower_txs[borrower_txs['type'] == 'debit'].nlargest(5, 'amount')
        
        return top_events.to_dict('records') + top_transactions.to_dict('records')

    def retrieve(self, query: str, borrower_id: str):
        print("\n--- Phase 2b: Retrieving Evidence ---")
        dense_results = self._dense_retrieval(query)
        sparse_results = self._sparse_retrieval(borrower_id)
        
        # Combine and format results into a list of strings
        evidence = []
        for item in dense_results:
            evidence.append(f"Communication Record: On {item['comm_date']}, the borrower communicated via {item['comm_channel']}: '{item['comm_text']}'")
        for item in sparse_results:
            if 'event_type' in item:
                evidence.append(f"Event Record: An event of type '{item['event_type']}' with severity '{item['severity']}' occurred on {item['event_date']}.")
            elif 'amount' in item:
                evidence.append(f"Transaction Record: A debit of {item['amount']:.2f} was made for '{item['description']}' on {item['date']}.")
        
        print(f"Retrieved {len(evidence)} initial pieces of evidence.")
        return evidence

# --- UPGRADE: PHASE 3A - POST-RETRIEVAL RE-RANKING ---

class ReRanker:
    """Uses a Cross-Encoder to re-rank retrieved evidence for relevance."""
    def __init__(self):
        print("\n--- Phase 3a: Initializing Re-ranker ---")
        # Cross-encoders are more accurate for ranking than sentence transformers
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("Re-ranker initialized.")

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[str]:
        print(f"Re-ranking {len(documents)} pieces of evidence...")
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        
        # Combine documents with scores and sort
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top_k most relevant documents
        ranked_docs = [doc for doc, score in doc_scores[:top_k]]
        print(f"Re-ranked and selected top {len(ranked_docs)} pieces of evidence.")
        return ranked_docs

# --- PHASE 3B: NARRATIVE GENERATION ---

def generate_narrative(query: str, evidence: list[str]):
    """Generates the final human-readable narrative using an LLM."""
    print("\n--- Phase 3b: Generating Final Narrative ---")
    
    # Format the evidence for the prompt
    formatted_evidence = "\n".join([f"- {item}" for item in evidence])
    
    prompt_template = """
    You are an expert credit risk analyst for a retail bank. Your task is to provide a concise, clear, and actionable risk summary for a borrower.
    Synthesize the provided evidence into a single paragraph. Do not just list the evidence; explain what it means in the context of credit risk.
    Your final output must be a single paragraph.

    QUERY: {query}

    AVAILABLE EVIDENCE:
    {evidence}

    RISK NARRATIVE:
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["query", "evidence"])
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    print("Sending request to LLM...")
    result = chain.run(query=query, evidence=formatted_evidence)
    return result


# --- MAIN EXECUTION ORCHESTRATOR ---
def run_explainability_pipeline(borrower_id: str):
    """Runs the full, upgraded RAG pipeline for a given borrower."""
    print(f"\n{'='*50}")
    print(f"STARTING AURA EXPLAINABILITY PIPELINE FOR: {borrower_id}")
    print(f"{'='*50}")

    try:
        comms_df = pd.read_csv(COMMUNICATIONS_PATH)
        events_df = pd.read_csv(EVENTS_PATH)
        transactions_df = pd.read_csv(TRANSACTIONS_PATH)
        borrowers_df = pd.read_csv(BORROWERS_ENHANCED_PATH)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure all data files exist.")
        return

    # Phase 1
    augmented_query = augment_query(borrower_id, events_df, borrowers_df)
    
    # Phase 2
    retriever = HybridRetriever(comms_df, events_df, transactions_df)
    initial_evidence = retriever.retrieve(augmented_query, borrower_id)
    
    # UPGRADE: Phase 3a
    reranker = ReRanker()
    ranked_evidence = reranker.rerank(augmented_query, initial_evidence)
    
    # Phase 3b
    final_narrative = generate_narrative(augmented_query, ranked_evidence)
    
    print(f"\n{'='*50}")
    print("PIPELINE COMPLETE. FINAL OUTPUT:")
    print(f"{'='*50}")
    print(final_narrative)

if __name__ == "__main__":
    # This block allows the script to be run directly for testing.
    # The run_pipeline.py script will automatically find the highest-risk borrower.
    print("Running explainability_engine.py directly for testing...")
    
    # Check if an API key is available
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n[FATAL ERROR] GOOGLE_API_KEY is not set. Please create a .env file and run 'test_api_key.py' to verify.")
    else:
        try:
            # Find the highest-risk borrower from the last run
            scores_df = pd.read_csv(AURA_SCORES_PATH)
            if not scores_df.empty:
                # To test a real flag, we use the What-If analysis target
                # In a real run, this would be the highest score from the model
                high_risk_borrower = "borrower_018" # The one we engineered for a late-stage crisis
                run_explainability_pipeline(high_risk_borrower)
            else:
                print("aura_risk_scores.csv is empty. Running with a default test case.")
                run_explainability_pipeline("borrower_022") # A default complex case
        except FileNotFoundError:
            print(f"Could not find {AURA_SCORES_PATH}. Please run the full pipeline first.")
            print("Running with a default test case.")
            run_explainability_pipeline("borrower_022")
