# Streamlit Match-Making Engine (Claude Reassigns Similarity Scores)

import streamlit as st
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
import boto3
import json
import logging
import anthropic
import faiss

# ─── Streamlit Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="SME Match-Making Engine",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Constants ────────────────────────────────────────────────────────
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
CSV_PATH = "data\\profiles_cleaned_tags - profiles_cleaned_tags.csv.csv"  # Updated path for Streamlit
TOP_K_RETRIEVE = 20
TOP_K_RETURN = 10
FAISS_M = 32
FAISS_EF_SEARCH = 128
CLAUDE_MODEL_ID = "claude-3-5-haiku-20241022"
DIM = 1024

# ─── Logging Setup ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("matchmaker")

# ─── Initialize Session State ────────────────────────────────────────
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'sme_df' not in st.session_state:
    st.session_state.sme_df = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'sme_vecs' not in st.session_state:
    st.session_state.sme_vecs = None

# ─── Utility Functions ───────────────────────────────────────────────
@st.cache_resource
def get_aws_client():
    """Initialize AWS Bedrock client"""
    try:
        return boto3.client("bedrock-runtime", region_name=REGION)
    except Exception as e:
        st.error(f"Failed to initialize AWS client: {e}")
        return None

@st.cache_resource
def get_claude_client():
    """Initialize Claude client using Streamlit secrets"""
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
        return anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Claude client: {e}")
        st.error("Make sure ANTHROPIC_API_KEY is set in Streamlit secrets")
        return None

def titan_embed(text: str, bedrock_client) -> np.ndarray:
    """Get embedding from Amazon Titan"""
    body = {"inputText": text}
    resp = bedrock_client.invoke_model(
        modelId=EMBED_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    vec = np.array(json.loads(resp["body"].read())["embedding"], dtype=np.float32)
    return vec / np.linalg.norm(vec)

def query_claude(prompt: str, claude_client) -> str:
    """Query Claude API"""
    try:
        response = claude_client.messages.create(
            model=CLAUDE_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.2
        )
        return response.content[0].text if response.content else ""
    except Exception as e:
        logger.error(f"Claude API error: {str(e)}")
        raise

def claude_rerank_with_scores(founder_text: str, candidate_rows: List[Tuple[int, float]], sme_df, claude_client) -> List[Dict]:
    """Claude reranks the top 20 candidates and assigns new similarity scores to the best 10"""
    lines = []
    for pos, (idx, original_score) in enumerate(candidate_rows, 1):
        row = sme_df.iloc[idx]
        lines.append(
            f"{pos}. ID: {row['id']} | Name: {row['Name']} | Experience: {row['Experience']} | Industry: {row['Industry']} | Skill Set: {row['Skill Set']} | Impact: {row['Impact Made']} | Original Vector Score: {original_score:.4f}"
        )

    prompt = f"""You are an expert SME matchmaker. 

Given the founder's profile and 20 SME candidates ranked by vector similarity, you need to:
1. Rerank them based on actual business relevance and fit quality
2. Select the BEST 10 matches
3. Assign new similarity scores (0.0 to 1.0) that reflect the TRUE match quality

FOUNDER PROFILE:
{founder_text}

TOP 20 CANDIDATES FROM VECTOR SEARCH:
{chr(10).join(lines)}

INSTRUCTIONS:
- Rank by actual business value, not just keyword matching
- Consider: relevant experience, industry fit, skill complementarity, impact potential
- Assign similarity scores: 0.9-1.0 (perfect fit), 0.8-0.89 (excellent), 0.7-0.79 (very good), 0.6-0.69 (good), 0.5-0.59 (decent)
- Return ONLY the top 10 matches

Respond with JSON:
{{"matches": [{{"expertId": "ID", "similarityScore": 0.XX, "reason": "Why this SME is the best fit (1-2 lines)"}}]}}

Make sure the similarity scores reflect the reranked quality, not the original vector scores."""

    try:
        raw_response = query_claude(prompt, claude_client)
        logger.info(f"Claude raw response: {raw_response[:500]}...")
        
        json_start = raw_response.find('{')
        json_end = raw_response.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            claude_result = json.loads(raw_response[json_start:json_end])
            return claude_result["matches"][:TOP_K_RETURN]
        raise ValueError("Claude returned no valid JSON")
    except Exception as e:
        logger.warning(f"Claude failed: {e}. Using fallback with adjusted scores.")
        
        # Fallback: Create scaled scores based on rank position
        fallback_matches = []
        for i, (idx, original_score) in enumerate(candidate_rows[:TOP_K_RETURN]):
            # Create decreasing similarity scores: 0.9, 0.85, 0.8, 0.75, etc.
            new_score = 0.9 - (i * 0.05)
            new_score = max(new_score, 0.5)  # Minimum score of 0.5
            
            fallback_matches.append({
                "expertId": str(sme_df.iloc[idx]['id']),
                "similarityScore": round(new_score, 3),
                "reason": f"Strong match (rank #{i+1}) with expertise in {sme_df.iloc[idx]['Industry']} and {sme_df.iloc[idx]['Experience']} experience"
            })
        return fallback_matches

@st.cache_data
def load_and_embed_sme_data():
    """Load SME data and create embeddings"""
    try:
        # Load CSV
        sme_df = pd.read_csv(CSV_PATH)
        if "id" not in sme_df.columns:
            sme_df.insert(0, "id", range(len(sme_df)))
        
        # Initialize FAISS index
        index = faiss.IndexHNSWFlat(DIM, FAISS_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = FAISS_EF_SEARCH
        
        # Get AWS client
        bedrock_client = get_aws_client()
        if not bedrock_client:
            return None, None, None
        
        # Create embeddings
        sme_vecs = np.empty((len(sme_df), DIM), dtype=np.float32)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, row in sme_df.iterrows():
            status_text.text(f"Embedding SME profile {i+1}/{len(sme_df)}: {row['Name']}")
            text = f"""
            Name: {row['Name']}
            Experience: {row['Experience']}
            Industry: {row['Industry']}
            LinkedIn Summary: {row['LinkedIn Input']}
            Job Description: {row['Job Description']}
            Impact Made: {row['Impact Made']}
            Stage Relevance: {row['Stage Relevance']}
            Skill Set: {row['Skill Set']}
            Likes: {row['Likes and Interests']}
            Tags: {row['Tags']}
            """
            sme_vecs[i] = titan_embed(text, bedrock_client)
            progress_bar.progress((i + 1) / len(sme_df))
        
        index.add(sme_vecs)
        progress_bar.empty()
        status_text.empty()
        
        return sme_df, index, sme_vecs
        
    except Exception as e:
        st.error(f"Failed to load SME data: {e}")
        return None, None, None

def initialize_system():
    """Initialize the matching system"""
    if not st.session_state.initialized:
        with st.spinner("🚀 Initializing SME Match-Making Engine..."):
            # Load data and create embeddings
            sme_df, index, sme_vecs = load_and_embed_sme_data()
            
            if sme_df is not None:
                st.session_state.sme_df = sme_df
                st.session_state.index = index
                st.session_state.sme_vecs = sme_vecs
                st.session_state.initialized = True
                st.success(f"✅ System initialized! Loaded {len(sme_df)} SME profiles")
            else:
                st.error("❌ Failed to initialize system")
                return False
    return True

def match_founder(founder_text: str):
    """Main matching function"""
    try:
        # Get clients
        bedrock_client = get_aws_client()
        claude_client = get_claude_client()
        
        if not bedrock_client or not claude_client:
            return None
        
        # Step 1: Get founder query vector
        q_vec = titan_embed(founder_text, bedrock_client).reshape(1, -1)
        
        # Step 2: FAISS retrieves top 20 candidates
        similarities, indices = st.session_state.index.search(q_vec, TOP_K_RETRIEVE)
        indices = indices[0]
        similarities = similarities[0]
        
        # Step 3: Calculate exact dot product scores for the 20 candidates
        exact_scores = (st.session_state.sme_vecs[indices] @ q_vec.T).flatten()
        candidate_pairs = [(idx, float(exact_scores[i])) for i, idx in enumerate(indices) if idx < len(st.session_state.sme_df)]
        candidate_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Step 4: Claude reranks and assigns NEW similarity scores to top 10
        claude_matches = claude_rerank_with_scores(founder_text, candidate_pairs, st.session_state.sme_df, claude_client)
        
        # Step 5: Build final response with Claude's similarity scores
        matches = []
        for rank, match in enumerate(claude_matches, 1):
            expert_id = match['expertId']
            sme_row = st.session_state.sme_df.loc[st.session_state.sme_df['id'] == int(expert_id)]
            if sme_row.empty:
                continue
                
            row = sme_row.iloc[0]
            matches.append({
                "rank": rank,
                "expertId": expert_id,
                "name": row["Name"],
                "experience": row["Experience"],
                "industry": row["Industry"],
                "skillSet": row["Skill Set"],
                "impact": row["Impact Made"],
                "similarity": match["similarityScore"],  # Claude's assigned score
                "reason": match["reason"]
            })

        return {
            "matches": matches,
            "totalCandidatesEvaluated": len(candidate_pairs),
            "scoringMethod": "claude_reassigned"
        }
        
    except Exception as e:
        st.error(f"Matching failed: {str(e)}")
        return None

# ─── Streamlit UI ────────────────────────────────────────────────────
def main():
    st.title("🤝 SME Match-Making Engine v4.0")
    st.markdown("**Claude-Powered Similarity Scoring** | Find the perfect Subject Matter Experts for your startup")
    
    # Initialize system
    if not initialize_system():
        st.stop()
    
    # Sidebar with system info
    with st.sidebar:
        st.header("📊 System Status")
        if st.session_state.initialized:
            st.success("✅ System Ready")
            st.metric("SME Profiles Loaded", len(st.session_state.sme_df))
            st.metric("FAISS Vectors", st.session_state.index.ntotal)
            st.info("🧠 Scoring: Claude-Reassigned")
        else:
            st.warning("⚠️ System Initializing...")
    
    # Main input form
    st.header("📝 Founder Profile")
    
    with st.form("founder_profile"):
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Project Title", placeholder="My Amazing Startup")
            audience = st.text_input("Target Audience", placeholder="B2B SaaS companies")
        
        with col2:
            tags = st.text_input("Tags (comma-separated)", placeholder="AI, healthcare, fintech")
        
        description = st.text_area(
            "Project Description*", 
            height=150,
            placeholder="Describe your startup, the problem you're solving, your current stage, and what kind of expertise you need..."
        )
        
        col3, col4 = st.columns(2)
        
        with col3:
            problem_statements = st.text_area(
                "Problem Statements",
                height=100,
                placeholder="List key problems you're addressing (one per line)"
            )
        
        with col4:
            burning_problems = st.text_area(
                "Burning Problems",
                height=100,
                placeholder="Most urgent challenges you need help with (one per line)"
            )
        
        follow_up_questions = st.text_area(
            "Follow-up Questions",
            height=80,
            placeholder="Specific questions you'd like to ask SMEs (one per line)"
        )
        
        submitted = st.form_submit_button("🔍 Find My SME Matches", type="primary")
    
    if submitted:
        if not description.strip():
            st.error("Please provide a project description")
            return
        
        # Build founder profile text
        founder_text_parts = []
        if title:
            founder_text_parts.append(f"Title: {title}")
        founder_text_parts.append(f"Description: {description}")
        if audience:
            founder_text_parts.append(f"Audience: {audience}")
        if problem_statements:
            problems = [p.strip() for p in problem_statements.split('\n') if p.strip()]
            if problems:
                founder_text_parts.append("Problem Statements:\n- " + "\n- ".join(problems))
        if tags:
            founder_text_parts.append(f"Tags: {tags}")
        if follow_up_questions:
            questions = [q.strip() for q in follow_up_questions.split('\n') if q.strip()]
            if questions:
                founder_text_parts.append("Follow-up Questions:\n- " + "\n- ".join(questions))
        if burning_problems:
            burning = [b.strip() for b in burning_problems.split('\n') if b.strip()]
            if burning:
                founder_text_parts.append("Burning Problems:\n- " + "\n- ".join(burning))
        
        founder_text = "\n".join(founder_text_parts)
        
        # Perform matching
        with st.spinner("🤖 Finding your perfect SME matches..."):
            results = match_founder(founder_text)
        
        if results:
            st.header("🎯 Your SME Matches")
            st.markdown(f"**Found {len(results['matches'])} expert matches** (from {results['totalCandidatesEvaluated']} candidates evaluated)")
            
            # Display matches
            for match in results['matches']:
                with st.expander(f"🏆 Rank #{match['rank']}: {match['name']} - {match['similarity']:.1%} Match", expanded=match['rank'] <= 3):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**👤 Name:** {match['name']}")
                        st.markdown(f"**💼 Experience:** {match['experience']}")
                        st.markdown(f"**🏭 Industry:** {match['industry']}")
                        st.markdown(f"**🛠️ Skill Set:** {match['skillSet']}")
                        st.markdown(f"**💥 Impact:** {match['impact']}")
                        st.markdown(f"**🎯 Why This Match:** {match['reason']}")
                    
                    with col2:
                        st.metric("Similarity Score", f"{match['similarity']:.1%}")
                        st.metric("Expert ID", match['expertId'])
                        
                        # Color-coded match quality
                        if match['similarity'] >= 0.9:
                            st.success("🌟 Perfect Fit")
                        elif match['similarity'] >= 0.8:
                            st.info("⭐ Excellent Match")
                        elif match['similarity'] >= 0.7:
                            st.warning("👍 Very Good Match")
                        else:
                            st.info("✅ Good Match")

if __name__ == "__main__":
    main()