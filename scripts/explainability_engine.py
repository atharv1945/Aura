import os
import sys
import time
import faiss
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- LangChain (v0.3+) imports ---
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# --- CONFIGURATION ---
# Assume scripts are run from the project root directory
# Get the absolute path of the directory containing the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go one level up to get the project root
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
COMMUNICATIONS_PATH = os.path.join(DATA_DIR, "communications.csv")
EVENTS_PATH = os.path.join(DATA_DIR, "events.csv")
TRANSACTIONS_PATH = os.path.join(DATA_DIR, "transactions.csv")
BORROWERS_ENHANCED_PATH = os.path.join(PROCESSED_DIR, "borrowers_enhanced.csv")
AURA_SCORES_PATH = os.path.join(PROCESSED_DIR, "aura_risk_scores.csv")

# Use config file for threshold if available, otherwise fallback
RISK_THRESHOLD = 0.6 # Default threshold
try:
    # Temporarily add script directory to path to import config
    sys.path.insert(0, SCRIPT_DIR)
    import data_config as config
    RISK_THRESHOLD = config.RISK_THRESHOLD # Overwrite default if found
    print(f"Loaded RISK_THRESHOLD={RISK_THRESHOLD} from data_config.py")
except (ImportError, AttributeError, FileNotFoundError):
    print(f"Warning: Could not import RISK_THRESHOLD from data_config. Using default {RISK_THRESHOLD}.")
finally:
    # Clean up sys.path modification
    if SCRIPT_DIR in sys.path:
        try:
            sys.path.remove(SCRIPT_DIR)
        except ValueError:
            pass # Path might have already been removed

# Load environment variables (for GOOGLE_API_KEY) from project root .env
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded environment variables from {dotenv_path}")
else:
    print(f"Warning: .env file not found at {dotenv_path}. GOOGLE_API_KEY should be set externally.")


# --- PHASE 1: PRE-RETRIEVAL (Query Augmentation) ---

def augment_query(borrower_id: str, events_df: pd.DataFrame, borrowers_df: pd.DataFrame) -> str:
    """Augments a simple query with specific context about the borrower."""
    print(f"\n--- Phase 1: Augmenting Query for {borrower_id} ---")

    # Safely get borrower profile
    borrower_profile = borrowers_df[borrowers_df['borrower_id'] == borrower_id]
    if borrower_profile.empty:
        print(f"Warning: Borrower profile not found for {borrower_id}.")
        return f"Analyze the overall credit risk for {borrower_id}, considering any available transaction or communication data."

    # Get archetype safely, provide default
    archetype = borrower_profile['behavioral_archetype'].iloc[0] if 'behavioral_archetype' in borrower_profile.columns else 'Unknown'

    # Get events safely
    borrower_events = events_df[events_df['borrower_id'] == borrower_id].copy()
    if borrower_events.empty:
        print("No events found for this borrower.")
        return f"Analyze the overall credit risk for {borrower_id} based on their profile as a '{archetype}'."

    # Ensure event_date is datetime and handle potential errors during conversion
    borrower_events['event_date'] = pd.to_datetime(borrower_events['event_date'], errors='coerce')
    borrower_events.dropna(subset=['event_date'], inplace=True) # Drop rows where date conversion failed
    if borrower_events.empty:
         print("No valid events with dates found after cleaning.")
         return f"Analyze the overall credit risk for {borrower_id} based on their profile as a '{archetype}'."

    # Calculate severity score safely
    borrower_events['severity_score'] = borrower_events['severity'].map({'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}).fillna(0).astype(int)

    # Get the 2 most recent severe events within the last 90 days for relevance
    now = datetime.now() # Use consistent 'now' timestamp
    recent_events = borrower_events[borrower_events['event_date'] > (now - timedelta(days=90))]

    if recent_events.empty or recent_events['severity_score'].max() < 3: # Check if any High/Critical events exist recently
        # Fallback: get any 2 most recent events if no severe ones in last 90 days
        top_events = borrower_events.sort_values(by='event_date', ascending=False).head(2)
        print("Warning: No High/Critical events in the last 90 days, using the 2 most recent events overall.")
    else:
        # Sort recent events first by severity (desc), then by date (desc) to get most critical recent
        top_events = recent_events.sort_values(by=['severity_score', 'event_date'], ascending=[False, False]).head(2)

    event_str = "no specific critical events found recently"
    if not top_events.empty:
        # Ensure event_type exists and handle potential NaNs safely
        top_events['event_type_str'] = top_events['event_type'].astype(str).fillna('Unknown Event')
        event_str = " and ".join(top_events['event_type_str'].unique().tolist()) # Use unique in case they are the same type

    augmented_query = f"Analyze the credit risk for {borrower_id}, focusing on recent critical events like '{event_str}' and considering their behavioral profile as a '{archetype}'."
    print(f"Augmented Query: \"{augmented_query}\"")
    return augmented_query

# --- PHASE 2: HYBRID RETRIEVAL ---

class HybridRetriever:
    """Retrieves evidence using both dense (semantic) and sparse (keyword) search."""
    def __init__(self, comms_df, events_df, transactions_df):
        print("\n--- Phase 2a: Initializing Hybrid Retriever ---")
        self.events_df = events_df.copy()
        self.transactions_df = transactions_df.copy()
        self.comms_df = comms_df.copy()
        self.index = None
        self.model = None
        self.faiss_to_df_index = None

        print("Loading sentence transformer model for dense retrieval...")
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2') # Standard effective model
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            print("Dense retrieval will be skipped.")

        if self.model and not self.comms_df.empty:
            print("Building FAISS index for communications data...")
            # Ensure comm_text is string, handle NaNs, and filter empty strings
            self.comms_df['comm_text'] = self.comms_df['comm_text'].astype(str).fillna('')
            valid_comms = self.comms_df[self.comms_df['comm_text'].str.strip() != '']

            if not valid_comms.empty:
                try:
                    print(f"Encoding {len(valid_comms)} communication texts...")
                    comms_embeddings = self.model.encode(valid_comms['comm_text'].tolist(), show_progress_bar=True, convert_to_tensor=True)
                    embedding_dim = comms_embeddings.shape[1]
                    print(f"Embeddings generated with dimension {embedding_dim}.")
                    self.index = faiss.IndexFlatL2(embedding_dim) # Simple index for moderate size
                    self.index.add(comms_embeddings.cpu().detach().numpy())
                    self.faiss_to_df_index = valid_comms.index # Store mapping
                    print(f"FAISS index built with {self.index.ntotal} vectors.")
                except Exception as e:
                    print(f"Error building FAISS index: {e}")
                    self.index = None # Disable dense search if index fails
            else:
                print("Warning: Communications data is empty or contains no text after cleaning. Dense retrieval skipped.")
        elif self.comms_df.empty:
             print("Warning: Communications dataframe is empty. Dense retrieval skipped.")
        else:
             print("Skipping FAISS index build because SentenceTransformer model failed to load.")

        print("Retriever initialized.")

    def _dense_retrieval(self, query: str, k: int = 5):
        """Performs dense vector search."""
        if self.index is None or self.index.ntotal == 0 or self.model is None:
             print("Skipping dense retrieval (index/model unavailable or index empty).")
             return []
        try:
            print(f"Performing dense search for query: '{query[:50]}...'")
            query_embedding = self.model.encode([query])
            actual_k = min(k, self.index.ntotal)
            if actual_k <= 0: return [] # Cannot search index with 0 vectors

            distances, indices = self.index.search(query_embedding, actual_k)

            # Check if indices are valid before attempting to map
            valid_indices_mask = indices[0] != -1 # FAISS uses -1 for no result
            valid_faiss_indices = indices[0][valid_indices_mask]

            if len(valid_faiss_indices) == 0:
                 print("Dense search returned no valid indices.")
                 return []

            # Map FAISS indices back to original DataFrame indices
            # Ensure faiss_to_df_index is not None and indices are within bounds
            if self.faiss_to_df_index is None or max(valid_faiss_indices) >= len(self.faiss_to_df_index):
                 print("Error: FAISS index mapping is invalid.")
                 return []
            original_indices = self.faiss_to_df_index[valid_faiss_indices]

            # Ensure indices exist in the original dataframe (safety check)
            valid_original_indices = original_indices[original_indices.isin(self.comms_df.index)]
            if len(valid_original_indices) == 0:
                 print("Dense search indices not found in original DataFrame.")
                 return []

            results = self.comms_df.loc[valid_original_indices].to_dict('records')
            print(f"Dense search returned {len(results)} results.")
            return results
        except Exception as e:
            print(f"Error during dense retrieval search: {e}")
            return []

    def _sparse_retrieval(self, borrower_id: str):
        """Performs targeted retrieval from structured event and transaction data."""
        sparse_evidence = []
        now = datetime.now() # Consistent timestamp

        # Retrieve recent critical events
        try:
            borrower_events = self.events_df[self.events_df['borrower_id'] == borrower_id].copy()
            if not borrower_events.empty:
                 borrower_events['event_date'] = pd.to_datetime(borrower_events['event_date'], errors='coerce')
                 borrower_events.dropna(subset=['event_date'], inplace=True)
                 if not borrower_events.empty:
                     borrower_events['severity_score'] = borrower_events['severity'].map({'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}).fillna(0).astype(int)
                     # Get top 3 most recent, highest severity events in last 180 days
                     recent_events = borrower_events[borrower_events['event_date'] > (now - timedelta(days=180))]
                     # Sort by severity first, then date
                     top_events = recent_events.sort_values(by=['severity_score', 'event_date'], ascending=[False, False]).head(3)
                     sparse_evidence.extend(top_events.to_dict('records'))
        except Exception as e:
             print(f"Warning: Error retrieving events for sparse search: {e}")

        # Retrieve recent large debit transactions
        try:
            borrower_txs = self.transactions_df[self.transactions_df['borrower_id'] == borrower_id].copy()
            if not borrower_txs.empty:
                 borrower_txs['date'] = pd.to_datetime(borrower_txs['date'], errors='coerce')
                 borrower_txs.dropna(subset=['date'], inplace=True)
                 # Get top 5 largest debit transactions in the last 90 days
                 recent_txs = borrower_txs[borrower_txs['date'] > (now - timedelta(days=90))]
                 top_transactions = recent_txs[recent_txs['type'] == 'debit'].nlargest(5, 'amount')
                 sparse_evidence.extend(top_transactions.to_dict('records'))
        except Exception as e:
             print(f"Warning: Error retrieving transactions for sparse search: {e}")

        print(f"Sparse search found {len(sparse_evidence)} items (events + transactions).")
        return sparse_evidence


    def retrieve(self, query: str, borrower_id: str, k_dense: int = 7):
        """Combines dense and sparse results."""
        print("\n--- Phase 2b: Retrieving Evidence ---")
        dense_results = self._dense_retrieval(query, k=k_dense)
        sparse_results = self._sparse_retrieval(borrower_id)

        evidence_set = set() # Use set for automatic deduplication

        # Process Dense Results (Communications)
        for item in dense_results:
            # Safely get and format data
            comm_date_obj = pd.to_datetime(item.get('comm_date', pd.NaT), errors='coerce')
            comm_date = comm_date_obj.strftime('%Y-%m-%d') if pd.notna(comm_date_obj) else 'Unknown Date'
            channel = str(item.get('comm_channel', 'N/A')).strip()
            text = str(item.get('comm_text', '')).strip()
            if text:
                evidence_set.add(f"Communication ({comm_date}, {channel}): {text}")

        # Process Sparse Results (Events and Transactions)
        for item in sparse_results:
            if 'event_type' in item: # Check if it looks like an event
                event_type = str(item.get('event_type', 'N/A')).strip()
                severity = str(item.get('severity', 'N/A')).strip()
                event_date_obj = pd.to_datetime(item.get('event_date', pd.NaT), errors='coerce') # Coerce errors here too
                date_str = event_date_obj.strftime('%Y-%m-%d') if pd.notna(event_date_obj) else 'Unknown Date'
                evidence_set.add(f"Event ({date_str}): Type='{event_type}', Severity='{severity}'.")
            elif 'amount' in item and 'date' in item: # Check if it looks like a transaction
                amount = item.get('amount', 0)
                # Ensure amount is numeric before formatting
                try:
                    amount_float = float(amount)
                except (ValueError, TypeError):
                    amount_float = 0 # Default if conversion fails

                desc = str(item.get('description', 'N/A')).strip()
                tx_date_obj = pd.to_datetime(item.get('date', pd.NaT), errors='coerce')
                date_str = tx_date_obj.strftime('%Y-%m-%d %H:%M') if pd.notna(tx_date_obj) else 'Unknown Date'
                # Only add meaningful transactions
                if amount_float > 0:
                     evidence_set.add(f"Transaction ({date_str}): Debit of {amount_float:.2f} for '{desc}'.")

        evidence = sorted(list(evidence_set)) # Sort for consistency
        print(f"Retrieved {len(evidence)} unique initial pieces of evidence.")
        return evidence


# --- PHASE 3A: POST-RETRIEVAL RE-RANKING ---

class ReRanker:
    """Uses a Cross-Encoder to re-rank retrieved evidence for relevance."""
    def __init__(self):
        print("\n--- Phase 3a: Initializing Re-ranker ---")
        self.model = None
        try:
            # Using a well-regarded cross-encoder for relevance ranking
            self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
            print("Re-ranker initialized.")
        except Exception as e:
            print(f"Error initializing CrossEncoder model: {e}. Download might be needed or model name incorrect.")
            print("Re-ranking will be skipped.")

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[str]:
        """Re-ranks documents based on query relevance."""
        if not self.model or not documents:
             print("Skipping re-ranking (model unavailable or no documents).")
             # Return top k based on original retrieval order as a fallback
             return documents[:top_k]

        print(f"Re-ranking {len(documents)} pieces of evidence for query: '{query[:50]}...'")
        # Prepare pairs for the cross-encoder: [query, document]
        pairs = [[query, doc] for doc in documents]
        try:
            # Predict relevance scores
            scores = self.model.predict(pairs, show_progress_bar=True)

            # Combine documents with scores and sort descending by score
            doc_scores = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

            # Select the top_k most relevant documents
            ranked_docs = [doc for doc, score in doc_scores[:top_k]]
            print(f"Re-ranked and selected top {len(ranked_docs)} pieces of evidence.")
            # # Optional: Print scores for debugging
            # print("Top evidence scores:")
            # for doc, score in doc_scores[:top_k]:
            #      print(f"  Score: {score:.4f} | Doc: {doc[:100]}...")
            return ranked_docs
        except Exception as e:
            print(f"Error during re-ranking prediction: {e}")
            print("Returning original top K documents as fallback.")
            return documents[:top_k]

# --- PHASE 3B: NARRATIVE GENERATION ---

def generate_narrative(query: str, evidence: list[str]):
    """Generates the final human-readable narrative using an LLM."""
    print("\n--- Phase 3b: Generating Final Narrative ---")

    if not evidence:
        print("No evidence provided to LLM.")
        return "No specific evidence was retrieved to generate a detailed narrative. Please review the borrower's overall profile manually."

    # Ensure evidence list contains only non-empty strings
    formatted_evidence = "\n".join([f"- {str(item)}" for item in evidence if isinstance(item, str) and str(item).strip()])
    if not formatted_evidence:
         print("Formatted evidence is empty after cleaning. Cannot generate narrative.")
         return "Retrieved evidence was empty or invalid. Cannot generate narrative."

    # More specific instructions for the LLM
    prompt_template = """
    **Role:** You are an expert Credit Risk Analyst reviewing a borrower's file.
    **Task:** Based *only* on the evidence provided below, write a concise risk assessment narrative (one paragraph) suitable for a credit review meeting.
    **Instructions:**
    1.  **Synthesize, Don't List:** Explain the *implications* of the evidence (e.g., impact on repayment ability, signs of distress, potential fraud). Connect the dots between different pieces of evidence if possible.
    2.  **Evidence Only:** Stick strictly to the information given in the 'AVAILABLE EVIDENCE'. Do not add external knowledge, opinions, or recommendations unless directly supported by the evidence.
    3.  **Conciseness:** Be brief and to the point. Aim for 3-5 key sentences summarizing the risk profile based on the evidence.
    4.  **Format:** Output must be a single paragraph. Start directly with the analysis (e.g., "The borrower exhibits signs of...").

    **ANALYST QUERY:** {query}

    **AVAILABLE EVIDENCE (Ranked by relevance):**
    {evidence}

    **RISK NARRATIVE (One Paragraph):**
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["query", "evidence"])

    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Cannot call LLM.")

        # Use a recommended, reliable Gemini model
        llm = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", # Use the full model path compatible with older v1beta API endpoints.
                                     temperature=0.2, # Lower temperature for factual summary
                                     google_api_key=api_key,
                                     convert_system_message_to_human=True) # Helps some models understand prompts better

        # Modern LCEL (LangChain Expression Language) approach
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        print(f"Sending request to Gemini model ({llm.model})...")
        start_time = time.time()
        # Use .invoke() for modern LCEL chains
        narrative = chain.invoke({"query": query, "evidence": formatted_evidence})
        end_time = time.time()
        print(f"LLM generation took {end_time - start_time:.2f} seconds.")

        # Basic cleanup - remove potential markdown/headers
        narrative = narrative.strip().replace("**RISK NARRATIVE:**", "").replace("RISK NARRATIVE:", "").strip()

    # Catch specific API errors if possible, otherwise general exception
    except Exception as e:
        print(f"ERROR during LLM generation: {type(e).__name__} - {e}")
        # Provide a more user-friendly error message
        error_msg = f"Failed to generate narrative due to an error: {str(e)[:150]}..."
        if "API key" in str(e):
            error_msg += " Please check if the GOOGLE_API_KEY is correct and active."
        elif "quota" in str(e).lower():
            error_msg += " API quota might have been exceeded."
        elif "content has been blocked" in str(e).lower():
             # Check if response object exists and has safety ratings
             # This part might need adjustment depending on the exact exception structure from langchain/google-generativeai
             # safety_info = getattr(e, 'response', {}).get('prompt_feedback', {}).get('safety_ratings', [])
             # if safety_info:
             #      error_msg += f" Generation failed due to content safety filters: {safety_info}"
             # else:
                  error_msg += " Generation failed due to content safety filters. The evidence might contain sensitive terms."

        elif "Deadline Exceeded" in str(e) or "504" in str(e):
             error_msg += " The request to the LLM timed out. You might try again later."
        elif "Model not found" in str(e):
             error_msg += " The specified LLM model name might be incorrect or unavailable."

        return error_msg


# --- MAIN EXECUTION ORCHESTRATOR ---
def run_explainability_pipeline(borrower_id: str):
    """Runs the full, upgraded RAG pipeline for a given borrower."""
    print(f"\n{'='*50}")
    print(f"STARTING AURA EXPLAINABILITY PIPELINE FOR: {borrower_id}")
    print(f"{'='*50}")

    try:
        # Load data with improved error handling and date parsing
        print(f"Loading data from raw dir: {DATA_DIR} and processed dir: {PROCESSED_DIR}...")

        # Define expected date columns and formats for robust parsing
        date_parsers = {
            COMMUNICATIONS_PATH: {'comm_date': '%Y-%m-%d'},
            EVENTS_PATH: {'event_date': '%Y-%m-%d'},
            TRANSACTIONS_PATH: {'date': '%Y-%m-%d %H:%M:%S'}
        }

        # Function to load and parse dates safely
        def load_and_parse(path, parse_dict):
            df = pd.read_csv(path)
            for col, fmt in parse_dict.items():
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                else:
                    print(f"Warning: Expected date column '{col}' not found in {os.path.basename(path)}.")
            return df

        comms_df = load_and_parse(COMMUNICATIONS_PATH, date_parsers[COMMUNICATIONS_PATH])
        events_df = load_and_parse(EVENTS_PATH, date_parsers[EVENTS_PATH])
        transactions_df = load_and_parse(TRANSACTIONS_PATH, date_parsers[TRANSACTIONS_PATH])
        borrowers_df = pd.read_csv(BORROWERS_ENHANCED_PATH)

        # --- Data Validation and Type Conversion ---
        required_files = {'comms': comms_df, 'events': events_df, 'tx': transactions_df, 'borrowers': borrowers_df}
        for name, df in required_files.items():
            if df.empty:
                print(f"Warning: {name}.csv loaded as empty DataFrame.")
            if 'borrower_id' not in df.columns:
                 print(f"Fatal Error: 'borrower_id' column missing in {name}.csv. Cannot proceed.")
                 return
            # Convert borrower_id to string AFTER loading
            df['borrower_id'] = df['borrower_id'].astype(str)

        # Drop rows with invalid dates after attempting parse
        comms_df.dropna(subset=['comm_date'], inplace=True)
        events_df.dropna(subset=['event_date'], inplace=True)
        transactions_df.dropna(subset=['date'], inplace=True)
        print("Data loaded and basic validation complete.")

    except FileNotFoundError as e:
        print(f"Fatal Error: Data file not found - {e}. Ensure all raw and processed files exist. Cannot proceed.")
        return
    except Exception as e:
        print(f"Fatal Error loading or parsing data: {e}. Check CSV formats and paths. Cannot proceed.")
        return

    # --- Run Pipeline Phases ---
    # Phase 1: Augment Query
    augmented_query = augment_query(borrower_id, events_df, borrowers_df)

    # Phase 2: Retrieve Evidence
    retriever = HybridRetriever(comms_df, events_df, transactions_df)
    initial_evidence = retriever.retrieve(augmented_query, borrower_id, k_dense=10) # Retrieve more initially

    # Phase 3a: Re-rank Evidence
    reranker = ReRanker()
    ranked_evidence = reranker.rerank(augmented_query, initial_evidence, top_k=5) # Select top 5-7 for LLM

    # Phase 3b: Generate Narrative
    final_narrative = generate_narrative(augmented_query, ranked_evidence)

    # --- Display Final Output ---
    print(f"\n{'='*50}")
    print(f"PIPELINE COMPLETE FOR {borrower_id}. FINAL NARRATIVE:")
    print(f"{'='*50}")
    print(final_narrative)

# --- Main execution logic ---
if __name__ == "__main__":
    print("Running explainability_engine.py...")

    # Check for API Key first
    api_key_present = os.getenv("GOOGLE_API_KEY")
    if not api_key_present:
        print("\n[FATAL ERROR] GOOGLE_API_KEY is not set.")
        print(f"Please ensure you have a .env file in the project root ({PROJECT_ROOT})")
        print("with the line: GOOGLE_API_KEY=\"YOUR_API_KEY_HERE\"")
        print("You can test your key using 'python scripts/test_api_key.py'")
        sys.exit(1) # Exit if key is missing

    flagged_borrowers = []
    try:
        # --- DYNAMICALLY FIND FLAGGED BORROWERS ---
        if os.path.exists(AURA_SCORES_PATH):
            scores_df = pd.read_csv(AURA_SCORES_PATH)
            if not scores_df.empty:
                scores_df['borrower_id'] = scores_df['borrower_id'].astype(str) # Ensure consistent type
                # Handle potential NaN scores before comparison
                scores_df['aura_risk_score'] = pd.to_numeric(scores_df['aura_risk_score'], errors='coerce')
                # Filter out NaNs before applying threshold
                scores_df.dropna(subset=['aura_risk_score'], inplace=True)
                flagged_borrowers_df = scores_df[scores_df['aura_risk_score'] > RISK_THRESHOLD]
                flagged_borrowers = flagged_borrowers_df['borrower_id'].tolist()
                print(f"\nFound {len(flagged_borrowers)} borrowers flagged by predictive engine (Score > {RISK_THRESHOLD}).")
            else:
                print(f"Warning: {AURA_SCORES_PATH} is empty.")
        else:
             print(f"Warning: Could not find {AURA_SCORES_PATH}. Cannot automatically determine flagged borrowers.")

    except pd.errors.EmptyDataError:
        print(f"Warning: {AURA_SCORES_PATH} is empty.")
    except Exception as e:
        print(f"Error reading or processing {AURA_SCORES_PATH}: {e}")

    # --- Run pipeline for flagged borrowers, or default if none ---
    if flagged_borrowers:
        print(f"Running Explainability Engine for flagged borrowers: {flagged_borrowers}")
        for borrower_id in flagged_borrowers:
            run_explainability_pipeline(borrower_id)
    else:
        # Define a default borrower known to have a complex history (our engineered case)
        default_borrower = "borrower_018"
        print(f"\nNo borrowers were flagged by the predictive engine in this run (or scores file missing/empty).")
        print(f"Running explainability pipeline for a default high-risk example ({default_borrower}) for demonstration.")
        # Ensure the default borrower exists in the borrowers file for a meaningful run
        try:
             # Check if borrowers_enhanced exists before trying to read
             if os.path.exists(BORROWERS_ENHANCED_PATH):
                 borrowers_check_df = pd.read_csv(BORROWERS_ENHANCED_PATH)
                 borrowers_check_df['borrower_id'] = borrowers_check_df['borrower_id'].astype(str)
                 if default_borrower in borrowers_check_df['borrower_id'].tolist():
                      run_explainability_pipeline(default_borrower)
                 else:
                      print(f"Warning: Default borrower {default_borrower} not found in {BORROWERS_ENHANCED_PATH}. Cannot run default example.")
             else:
                  print(f"Warning: Cannot check for default borrower, {BORROWERS_ENHANCED_PATH} not found.")
        except Exception as e:
             print(f"Error trying to run default example: {e}")

    print("\nExplainability engine script finished.")
