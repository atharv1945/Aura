import pandas as pd
import os
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
# --- Remove Langchain Imports if fully switching ---
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_google_genai import ChatGoogleGenerativeAI
# --- Add Native Google AI Import ---
import google.generativeai as genai # <-- ADD THIS LINE
# --- End ---
from dotenv import load_dotenv
import numpy as np
from datetime import datetime, timedelta
import sys
import time

# --- (Keep CONFIGURATION the same) ---
# Assume scripts are run from the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
COMMUNICATIONS_PATH = os.path.join(DATA_DIR, "communications.csv")
EVENTS_PATH = os.path.join(DATA_DIR, "events.csv")
TRANSACTIONS_PATH = os.path.join(DATA_DIR, "transactions.csv")
BORROWERS_ENHANCED_PATH = os.path.join(PROCESSED_DIR, "borrowers_enhanced.csv")
AURA_SCORES_PATH = os.path.join(PROCESSED_DIR, "aura_risk_scores.csv")

RISK_THRESHOLD = 0.6 # Default threshold
try:
    sys.path.insert(0, SCRIPT_DIR)
    # Check if data_config exists before importing
    if os.path.exists(os.path.join(SCRIPT_DIR, "data_config.py")):
        import data_config as config
        RISK_THRESHOLD = config.RISK_THRESHOLD # Overwrite default if found
        print(f"Loaded RISK_THRESHOLD={RISK_THRESHOLD} from data_config.py")
    else:
        print(f"Warning: data_config.py not found. Using default RISK_THRESHOLD={RISK_THRESHOLD}.")
except Exception as e:
    print(f"Warning: Could not import RISK_THRESHOLD from data_config: {e}. Using default {RISK_THRESHOLD}.")
finally:
    if SCRIPT_DIR in sys.path:
        try: sys.path.remove(SCRIPT_DIR)
        except ValueError: pass

dotenv_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded environment variables from {dotenv_path}")
else:
    print(f"Warning: .env file not found at {dotenv_path}. GOOGLE_API_KEY should be set externally.")


# --- (Keep augment_query, HybridRetriever, ReRanker classes the same) ---
# --- PHASE 1: PRE-RETRIEVAL (Query Augmentation) ---
def augment_query(borrower_id: str, events_df: pd.DataFrame, borrowers_df: pd.DataFrame) -> str:
    print(f"\n--- Phase 1: Augmenting Query for {borrower_id} ---")
    borrower_profile = borrowers_df[borrowers_df['borrower_id'] == borrower_id]
    if borrower_profile.empty:
        print(f"Warning: Borrower profile not found for {borrower_id}.")
        return f"Analyze the overall credit risk for {borrower_id}."
    archetype = borrower_profile['behavioral_archetype'].iloc[0] if 'behavioral_archetype' in borrower_profile.columns else 'Unknown'
    borrower_events = events_df[events_df['borrower_id'] == borrower_id].copy()
    if borrower_events.empty:
        print("No events found for this borrower."); return f"Analyze risk for {borrower_id} ({archetype})."
    borrower_events['event_date'] = pd.to_datetime(borrower_events['event_date'], errors='coerce')
    borrower_events.dropna(subset=['event_date'], inplace=True)
    if borrower_events.empty:
         print("No valid events with dates found."); return f"Analyze risk for {borrower_id} ({archetype})."
    borrower_events['severity_score'] = borrower_events['severity'].map({'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}).fillna(0).astype(int)
    now = datetime.now()
    recent_events = borrower_events[borrower_events['event_date'] > (now - timedelta(days=90))]
    if recent_events.empty or recent_events['severity_score'].max() < 3:
        top_events = borrower_events.sort_values(by='event_date', ascending=False).head(2)
        print("Warning: No High/Critical events recently, using latest 2 events.")
    else:
        top_events = recent_events.sort_values(by=['severity_score', 'event_date'], ascending=[False, False]).head(2)
    event_str = "no specific critical events found recently"
    if not top_events.empty:
        top_events['event_type_str'] = top_events['event_type'].astype(str).fillna('Unknown')
        event_str = " and ".join(top_events['event_type_str'].unique().tolist())
    augmented_query = f"Analyze the credit risk for {borrower_id}, focusing on recent critical events like '{event_str}' and considering their behavioral profile as a '{archetype}'."
    print(f"Augmented Query: \"{augmented_query}\"")
    return augmented_query

# --- PHASE 2: HYBRID RETRIEVAL ---
class HybridRetriever:
    def __init__(self, comms_df, events_df, transactions_df):
        print("\n--- Phase 2a: Initializing Hybrid Retriever ---")
        self.events_df = events_df.copy(); self.transactions_df = transactions_df.copy(); self.comms_df = comms_df.copy()
        self.index = None; self.model = None; self.faiss_to_df_index = None
        print("Loading sentence transformer model..."); t_start = time.time()
        try: self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e: print(f"Error loading SBERT: {e}. Dense retrieval skipped.")
        print(f"Model loaded in {time.time()-t_start:.2f}s")
        if self.model and not self.comms_df.empty:
            print("Building FAISS index..."); t_start = time.time()
            self.comms_df['comm_text'] = self.comms_df['comm_text'].astype(str).fillna('')
            valid_comms = self.comms_df[self.comms_df['comm_text'].str.strip() != '']
            if not valid_comms.empty:
                try:
                    comms_embeddings = self.model.encode(valid_comms['comm_text'].tolist(), show_progress_bar=False, convert_to_tensor=True)
                    embedding_dim = comms_embeddings.shape[1]; print(f"Encoded {len(valid_comms)} texts (dim: {embedding_dim}).")
                    self.index = faiss.IndexFlatL2(embedding_dim)
                    self.index.add(comms_embeddings.cpu().detach().numpy())
                    self.faiss_to_df_index = valid_comms.index; print(f"FAISS index built ({self.index.ntotal} vectors).")
                except Exception as e: print(f"Error building FAISS index: {e}"); self.index = None
            else: print("Warning: No valid text in comms data. Dense retrieval skipped.")
            print(f"Index built in {time.time()-t_start:.2f}s")
        elif self.comms_df.empty: print("Warning: Comms df empty. Dense retrieval skipped.")
        else: print("Skipping FAISS index build (model load failed).")
        print("Retriever initialized.")

    def _dense_retrieval(self, query: str, k: int = 5):
        if self.index is None or self.index.ntotal == 0 or self.model is None: return []
        try:
            query_embedding = self.model.encode([query]); actual_k = min(k, self.index.ntotal)
            if actual_k <= 0: return []
            distances, indices = self.index.search(query_embedding, actual_k)
            valid_indices_mask = indices[0] != -1; valid_faiss_indices = indices[0][valid_indices_mask]
            if len(valid_faiss_indices) == 0: return []
            if self.faiss_to_df_index is None or max(valid_faiss_indices) >= len(self.faiss_to_df_index): return []
            original_indices = self.faiss_to_df_index[valid_faiss_indices]
            valid_original_indices = original_indices[original_indices.isin(self.comms_df.index)]
            if len(valid_original_indices) == 0: return []
            return self.comms_df.loc[valid_original_indices].to_dict('records')
        except Exception as e: print(f"Error dense retrieval search: {e}"); return []

    def _sparse_retrieval(self, borrower_id: str):
        sparse_evidence = []; now = datetime.now()
        try: # Events
            borrower_events = self.events_df[self.events_df['borrower_id'] == borrower_id].copy()
            if not borrower_events.empty:
                 borrower_events['event_date'] = pd.to_datetime(borrower_events['event_date'], errors='coerce'); borrower_events.dropna(subset=['event_date'], inplace=True)
                 if not borrower_events.empty:
                     borrower_events['severity_score'] = borrower_events['severity'].map({'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}).fillna(0).astype(int)
                     recent_events = borrower_events[borrower_events['event_date'] > (now - timedelta(days=180))]
                     top_events = recent_events.sort_values(by=['severity_score', 'event_date'], ascending=[False, False]).head(3)
                     sparse_evidence.extend(top_events.to_dict('records'))
        except Exception as e: print(f"Warning: Error retrieving events: {e}")
        try: # Transactions
            borrower_txs = self.transactions_df[self.transactions_df['borrower_id'] == borrower_id].copy()
            if not borrower_txs.empty:
                 borrower_txs['date'] = pd.to_datetime(borrower_txs['date'], errors='coerce'); borrower_txs.dropna(subset=['date'], inplace=True)
                 recent_txs = borrower_txs[borrower_txs['date'] > (now - timedelta(days=90))]
                 top_transactions = recent_txs[recent_txs['type'] == 'debit'].nlargest(5, 'amount')
                 sparse_evidence.extend(top_transactions.to_dict('records'))
        except Exception as e: print(f"Warning: Error retrieving transactions: {e}")
        return sparse_evidence

    def retrieve(self, query: str, borrower_id: str, k_dense: int = 7):
        print("\n--- Phase 2b: Retrieving Evidence ---"); t_start = time.time()
        dense_results = self._dense_retrieval(query, k=k_dense)
        sparse_results = self._sparse_retrieval(borrower_id)
        evidence_set = set()
        for item in dense_results: # Comms
            comm_date_obj=pd.to_datetime(item.get('comm_date',pd.NaT),errors='coerce'); comm_date=comm_date_obj.strftime('%Y-%m-%d') if pd.notna(comm_date_obj) else 'Unknown'
            channel=str(item.get('comm_channel','N/A')).strip(); text=str(item.get('comm_text','')).strip()
            if text: evidence_set.add(f"Comm ({comm_date}, {channel}): {text}")
        for item in sparse_results: # Events & Txs
            if 'event_type' in item:
                event_type=str(item.get('event_type','N/A')).strip(); severity=str(item.get('severity','N/A')).strip()
                event_date_obj=pd.to_datetime(item.get('event_date',pd.NaT),errors='coerce'); date_str=event_date_obj.strftime('%Y-%m-%d') if pd.notna(event_date_obj) else 'Unknown'
                evidence_set.add(f"Event ({date_str}): Type='{event_type}', Severity='{severity}'.")
            elif 'amount' in item and 'date' in item:
                try: amount_float=float(item.get('amount',0))
                except: amount_float=0
                desc=str(item.get('description','N/A')).strip(); tx_date_obj=pd.to_datetime(item.get('date',pd.NaT),errors='coerce')
                date_str=tx_date_obj.strftime('%Y-%m-%d %H:%M') if pd.notna(tx_date_obj) else 'Unknown'
                if amount_float > 0: evidence_set.add(f"Tx ({date_str}): Debit {amount_float:.2f} for '{desc}'.")
        evidence = sorted(list(evidence_set)); print(f"Retrieved {len(evidence)} unique evidence pieces in {time.time()-t_start:.2f}s.")
        return evidence

# --- PHASE 3A: POST-RETRIEVAL RE-RANKING ---
class ReRanker:
    def __init__(self):
        print("\n--- Phase 3a: Initializing Re-ranker ---"); self.model = None; t_start = time.time()
        try:
            self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512); print(f"Re-ranker initialized in {time.time()-t_start:.2f}s.")
        except Exception as e: print(f"Error initializing CrossEncoder: {e}. Re-ranking skipped.")

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[str]:
        if not self.model or not documents: print("Skipping re-ranking."); return documents[:top_k]
        print(f"Re-ranking {len(documents)} evidence pieces..."); t_start = time.time()
        pairs = [[query, doc] for doc in documents]
        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
            doc_scores = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            ranked_docs = [doc for doc, score in doc_scores[:top_k]]; print(f"Selected top {len(ranked_docs)} evidence pieces in {time.time()-t_start:.2f}s.")
            return ranked_docs
        except Exception as e: print(f"Error during re-ranking: {e}. Returning original top K."); return documents[:top_k]

# --- PHASE 3B: NARRATIVE GENERATION (Native Google AI) ---

def generate_narrative(query: str, evidence: list[str]):
    """Generates the final human-readable narrative using the native google-generativeai library."""
    print("\n--- Phase 3b: Generating Final Narrative (Native Google AI) ---")

    if not evidence:
        print("No evidence provided to LLM."); return "No specific evidence retrieved..."
    formatted_evidence = "\n".join([f"- {str(item)}" for item in evidence if isinstance(item, str) and str(item).strip()])
    if not formatted_evidence:
         print("Formatted evidence empty after cleaning."); return "Retrieved evidence empty/invalid."

    full_prompt = f"""
    **Role:** You are an expert Credit Risk Analyst reviewing a borrower's file.
    **Task:** Based *only* on the evidence provided below, write a concise risk assessment narrative (one paragraph) suitable for a credit review meeting.
    **Instructions:**
    1.  **Synthesize, Don't List:** Explain the *implications* of the evidence (e.g., impact on repayment ability, signs of distress, potential fraud). Connect the dots between different pieces of evidence if possible.
    2.  **Evidence Only:** Stick strictly to the information given in the 'AVAILABLE EVIDENCE'. Do not add external knowledge, opinions, or recommendations unless directly supported by the evidence.
    3.  **Conciseness:** Be brief and to the point. Aim for 3-5 key sentences summarizing the risk profile based on the evidence.
    4.  **Format:** Output must be a single paragraph. Start directly with the analysis (e.g., "The borrower exhibits signs of...").

    **ANALYST QUERY:** {query}

    **AVAILABLE EVIDENCE (Ranked by relevance):**
    {formatted_evidence}

    **RISK NARRATIVE (One Paragraph):**
    """

    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")

        # Configure the native client
        genai.configure(api_key=api_key)

        # Select the model
        model_name = "gemini-2.5-pro" # Using the latest flash as it worked before
        model = genai.GenerativeModel(model_name)
        print(f"Sending request to Gemini model ({model_name}) via native library...")
        start_time = time.time()

        # Make the API call
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.2), # Control creativity
            # Add safety settings to be less restrictive if needed (adjust cautiously)
            # safety_settings={
            #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            # }
        )
        end_time = time.time()
        print(f"LLM generation took {end_time - start_time:.2f} seconds.")

        # --- FIX: Robust text extraction for native library ---
        narrative = ""
        try:
            # Check for blocking first
            if not response.candidates:
                 prompt_feedback = getattr(response, 'prompt_feedback', None)
                 block_reason = getattr(prompt_feedback, 'block_reason', 'Unknown') if prompt_feedback else 'Unknown'
                 safety_ratings = getattr(prompt_feedback, 'safety_ratings', []) if prompt_feedback else []
                 print(f"Warning: Prompt blocked by API. Reason: {block_reason}. Safety Ratings: {safety_ratings}")
                 return f"LLM prompt blocked due to safety filters (Reason: {block_reason}). Review evidence/prompt."

            # Check candidate finish reason
            candidate = response.candidates[0]
            finish_reason_enum = getattr(genai.types.Candidate, 'FinishReason', None) # Get Enum safely
            expected_stop = finish_reason_enum.STOP if finish_reason_enum else 1 # Default expected value if Enum not found

            if candidate.finish_reason != expected_stop:
                finish_reason_name = finish_reason_enum(candidate.finish_reason).name if finish_reason_enum else str(candidate.finish_reason)
                print(f"Warning: Generation finished reason: {finish_reason_name}")
                safety_ratings = getattr(candidate, 'safety_ratings', [])
                safety_detail = f" Safety Ratings: {safety_ratings}" if safety_ratings else ""
                # Return specific message based on reason if possible
                if finish_reason_name == "SAFETY":
                     return f"LLM generation stopped due to SAFETY filters.{safety_detail}"
                elif finish_reason_name == "MAX_TOKENS":
                     return f"LLM generation stopped due to reaching maximum output tokens."
                else:
                     return f"LLM generation stopped prematurely ({finish_reason_name}).{safety_detail}"

            # Extract text from parts
            if candidate.content.parts:
                narrative = candidate.content.parts[0].text
            else:
                narrative = "" # No parts returned

        except AttributeError as ae:
            # Handle potential changes in the response object structure
            print(f"Warning: Error parsing LLM response object (AttributeError: {ae}). Trying direct .text access.")
            try:
                narrative = response.text # Simpler access sometimes works
            except AttributeError:
                print("Error: Could not extract text from LLM response using standard methods.")
                narrative = "" # Give up if .text also fails
        except Exception as parse_e:
             print(f"Error parsing LLM response object: {parse_e}")
             narrative = ""
        # --- END FIX ---


        narrative = narrative.strip() # Basic cleanup

        if not narrative or narrative.lower() == 'none':
             print("Warning: LLM generation returned an empty or 'None' response.")
             return "LLM generation resulted in an empty narrative. Evidence might be insufficient or unclear."

        return narrative

    except Exception as e:
        # Keep similar error handling as before
        print(f"ERROR during LLM generation: {type(e).__name__} - {e}")
        error_msg = f"Failed to generate narrative: {str(e)[:150]}..."
        if "API key" in str(e): error_msg += " Check GOOGLE_API_KEY."
        elif "quota" in str(e).lower(): error_msg += " API quota exceeded?"
        elif "permission_denied" in str(e).lower() or "403" in str(e): error_msg += " Permission denied. Check API key/project."
        elif "Model not found" in str(e) or "is not found" in str(e): error_msg += f" Model '{model_name}' not found or unavailable."
        # Add specific error for Resource Exhausted (e.g., rate limiting)
        elif "Resource has been exhausted" in str(e) or "429" in str(e):
            error_msg += " API rate limit likely exceeded. Please wait and try again."
        return error_msg


# --- (Keep run_explainability_pipeline and __main__ block the same) ---
# --- MAIN EXECUTION ORCHESTRATOR ---
def run_explainability_pipeline(borrower_id: str):
    """Runs the full, upgraded RAG pipeline for a given borrower."""
    print(f"\n{'='*50}")
    print(f"STARTING AURA EXPLAINABILITY PIPELINE FOR: {borrower_id}")
    print(f"{'='*50}"); t_pipeline_start = time.time()
    try:
        print(f"Loading data from raw: {DATA_DIR}, processed: {PROCESSED_DIR}...")
        date_parsers={'comms': {'comm_date': '%Y-%m-%d'},'events': {'event_date': '%Y-%m-%d'},'tx': {'date': '%Y-%m-%d %H:%M:%S'}}
        def load_and_parse(path, parse_dict):
             if not os.path.exists(path): raise FileNotFoundError(f"Required: {path}")
             df = pd.read_csv(path); df_name = os.path.basename(path)
             for col, fmt in parse_dict.items():
                 if col in df.columns:
                     original_col = df[col].copy(); df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                     failed_mask = df[col].isna() & pd.notna(original_col)
                     if failed_mask.any():
                          # print(f"Info: Format '{fmt}' failed for {failed_mask.sum()} rows in '{col}' ({df_name}), inferring...") # Less verbose
                          with pd.option_context('mode.chained_assignment', None): # Suppress warning if possible
                              df.loc[failed_mask, col] = pd.to_datetime(original_col[failed_mask], errors='coerce', infer_datetime_format=True)
                 # else: print(f"Info: Date col '{col}' not found in {df_name}.") # Less verbose
             return df

        comms_df = load_and_parse(COMMUNICATIONS_PATH, date_parsers['comms'])
        events_df = load_and_parse(EVENTS_PATH, date_parsers['events'])
        transactions_df = load_and_parse(TRANSACTIONS_PATH, date_parsers['tx'])
        borrowers_df = pd.read_csv(BORROWERS_ENHANCED_PATH)

        required_files = {'comms': comms_df, 'events': events_df, 'tx': transactions_df, 'borrowers': borrowers_df}
        for name, df in required_files.items():
            if df.empty: print(f"Warning: {name}.csv loaded empty.")
            if 'borrower_id' not in df.columns: raise ValueError(f"'borrower_id' missing in {name}.csv.")
            df['borrower_id'] = df['borrower_id'].astype(str)

        date_cols_to_check = {'comms': 'comm_date', 'events': 'event_date', 'tx': 'date'}
        for name, col in date_cols_to_check.items():
             df = required_files[name]
             if col in df.columns:
                  initial_len = len(df); df.dropna(subset=[col], inplace=True)
                  # if len(df) < initial_len: print(f"Info: Dropped {initial_len - len(df)} rows from {name}.csv due to invalid date in '{col}'.") # Less verbose

        print("Data loading and validation complete.")

    except (FileNotFoundError, ValueError) as e: print(f"Fatal Error: {e}. Cannot proceed."); return
    except Exception as e: print(f"Fatal Error loading/parsing data: {e}. Cannot proceed."); return

    augmented_query = augment_query(borrower_id, events_df, borrowers_df)
    retriever = HybridRetriever(comms_df, events_df, transactions_df)
    initial_evidence = retriever.retrieve(augmented_query, borrower_id, k_dense=10)
    reranker = ReRanker()
    ranked_evidence = reranker.rerank(augmented_query, initial_evidence, top_k=7)
    final_narrative = generate_narrative(augmented_query, ranked_evidence)

    print(f"\n{'='*50}")
    print(f"PIPELINE COMPLETE FOR {borrower_id} in {time.time()-t_pipeline_start:.2f}s. FINAL NARRATIVE:")
    print(f"{'='*50}")
    print(final_narrative if final_narrative else "[No narrative generated or error occurred]")

if __name__ == "__main__":
    print("Running explainability_engine.py...")
    api_key_present = os.getenv("GOOGLE_API_KEY")
    if not api_key_present:
        print(f"\n[FATAL ERROR] GOOGLE_API_KEY not set. Check .env file in {PROJECT_ROOT}")
        sys.exit(1)

    flagged_borrowers = []
    try:
        if os.path.exists(AURA_SCORES_PATH):
            scores_df = pd.read_csv(AURA_SCORES_PATH)
            if not scores_df.empty:
                scores_df['borrower_id'] = scores_df['borrower_id'].astype(str)
                scores_df['aura_risk_score'] = pd.to_numeric(scores_df['aura_risk_score'], errors='coerce')
                scores_df.dropna(subset=['aura_risk_score'], inplace=True)
                flagged_borrowers_df = scores_df[scores_df['aura_risk_score'] > RISK_THRESHOLD]
                flagged_borrowers = flagged_borrowers_df['borrower_id'].tolist()
                print(f"\nFound {len(flagged_borrowers)} borrowers flagged (Score > {RISK_THRESHOLD}).")
            else: print(f"Warning: {AURA_SCORES_PATH} is empty.")
        else: print(f"Warning: Could not find {AURA_SCORES_PATH}. Cannot find flagged borrowers.")
    except Exception as e: print(f"Error reading {AURA_SCORES_PATH}: {e}")

    if flagged_borrowers:
        print(f"Running Explainability Engine for flagged borrowers: {flagged_borrowers}")
        limit_flagged = 3
        for borrower_id in flagged_borrowers[:limit_flagged]: run_explainability_pipeline(borrower_id)
        if len(flagged_borrowers) > limit_flagged: print(f"... and {len(flagged_borrowers) - limit_flagged} more.")
    else:
        default_borrower = "borrower_018"
        print(f"\nNo borrowers flagged. Running for default example ({default_borrower}).")
        try:
             if os.path.exists(BORROWERS_ENHANCED_PATH):
                 borrowers_check_df = pd.read_csv(BORROWERS_ENHANCED_PATH)
                 borrowers_check_df['borrower_id'] = borrowers_check_df['borrower_id'].astype(str)
                 if default_borrower in borrowers_check_df['borrower_id'].tolist(): run_explainability_pipeline(default_borrower)
                 else:
                      fallback_borrower = "borrower_028"
                      if fallback_borrower in borrowers_check_df['borrower_id'].tolist():
                           print(f"Warning: {default_borrower} not found. Using fallback {fallback_borrower}."); run_explainability_pipeline(fallback_borrower)
                      else: print(f"Warning: Defaults ({default_borrower}, {fallback_borrower}) not found. Cannot run default.")
             else: print(f"Warning: Cannot check default borrower, {BORROWERS_ENHANCED_PATH} not found.")
        except Exception as e: print(f"Error trying default example: {e}")

    print("\nExplainability engine script finished.")

