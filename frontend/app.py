import streamlit as st
import requests
import json

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="TrustAI - Misinformation Detection",
    page_icon="🔍",
    layout="wide"
)

def check_claim(claim: str):
    """Fact-check a claim using the API."""
    try:
        payload = {"claim": claim, "include_similar_claims": True, "include_llm_analysis": True}
        response = requests.post(f"{API_BASE_URL}/check_claim", json=payload, timeout=30)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            error_data = response.json() if response.content else {}
            return {"success": False, "error": error_data.get("detail", f"API error: {response.status_code}")}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Connection error: {str(e)}"}

def get_score_color(score):
    """Get color based on trust score."""
    if score >= 80:
        return "🟢"
    elif score >= 60:
        return "🟡"
    elif score >= 40:
        return "🟠"
    else:
        return "🔴"

def main():
    # Header
    st.title("🔍 TrustAI")
    st.subheader("Real-time Misinformation Detection System")
    
    # Check API health
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            st.success("✅ API is healthy and ready")
        else:
            st.error("❌ API is not responding properly")
    except:
        st.error("❌ Cannot connect to API. Make sure the backend is running on localhost:8000")
        st.stop()
    
    st.markdown("---")
    
    # Sidebar with examples
    with st.sidebar:
        st.header("💡 Example Claims")
        examples = [
            "Vaccines cause autism in children",
            "The Earth is flat", 
            "Climate change is caused by humans",
            "You only use 10% of your brain",
            "The Great Wall of China is visible from space",
            "Lightning never strikes the same place twice"
        ]
        
        selected_example = st.selectbox("Choose an example:", [""] + examples, index=0)
        
        st.markdown("---")
        st.header("📊 Trust Score Guide")
        st.markdown("""
        - 🟢 **80-100**: Highly credible
        - 🟡 **60-79**: Mostly credible  
        - 🟠 **40-59**: Questionable
        - 🔴 **20-39**: Likely false
        - 🔴 **0-19**: Highly unreliable
        """)
    
    # Input form
    with st.form("fact_check_form"):
        claim_input = st.text_area(
            "Enter your claim:",
            value=selected_example if selected_example else "",
            height=100,
            max_chars=1000,
            placeholder="e.g., Vaccines cause autism in children"
        )
        
        submitted = st.form_submit_button("🔍 Fact-Check", use_container_width=True)
        
        if claim_input:
            char_count = len(claim_input)
            if char_count < 10:
                st.warning(f"⚠️ Claim too short. Need at least 10 characters. Current: {char_count}")
            else:
                st.info(f"📝 Character count: {char_count}/1000")
    
    # Process claim
    if submitted and claim_input:
        if len(claim_input.strip()) < 10:
            st.error("❌ Claim must be at least 10 characters long")
        else:
            with st.spinner("🔄 Analyzing claim..."):
                result = check_claim(claim_input)
                
                if result["success"]:
                    data = result["data"]
                    
                    # Main results
                    st.markdown("## 📊 Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        score_emoji = get_score_color(data["trust_score"])
                        st.metric(
                            "Trust Score", 
                            f"{score_emoji} {data['trust_score']}/100",
                            help="How trustworthy this claim appears to be"
                        )
                    
                    with col2:
                        conf_emoji = get_score_color(data["confidence"])
                        st.metric(
                            "Confidence", 
                            f"{conf_emoji} {data['confidence']}/100",
                            help="How confident we are in this assessment"
                        )
                    
                    with col3:
                        st.metric(
                            "Processing Time", 
                            f"⚡ {data['processing_time']:.0f}ms",
                            help="Time taken to analyze the claim"
                        )
                    
                    # Explanation
                    st.markdown("### 📝 Explanation")
                    st.write(data["explanation"])
                    
                    # LLM Analysis
                    if data.get("llm_analysis"):
                        st.markdown("### 🧠 AI Analysis")
                        with st.expander("View detailed AI analysis", expanded=True):
                            st.write(data["llm_analysis"])
                    
                    # Similar Claims
                    if data.get("similar_claims") and len(data["similar_claims"]) > 0:
                        st.markdown(f"### 🔗 Similar Claims ({len(data['similar_claims'])})")
                        
                        for i, claim_data in enumerate(data["similar_claims"]):
                            with st.expander(f"📄 Similar Claim #{i+1}: {claim_data['verdict']}", expanded=False):
                                
                                # Verdict with color
                                verdict = claim_data['verdict']
                                if 'false' in verdict.lower():
                                    verdict_color = "🔴"
                                elif 'true' in verdict.lower() and 'false' not in verdict.lower():
                                    verdict_color = "🟢"
                                else:
                                    verdict_color = "🟡"
                                
                                st.markdown(f"**Verdict:** {verdict_color} {verdict}")
                                st.markdown(f"**Claim:** {claim_data['claim']}")
                                st.markdown(f"**Explanation:** {claim_data['explanation'][:300]}...")
                                
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.markdown(f"**Source:** {claim_data['source']}")
                                with col_b:
                                    st.markdown(f"**Category:** {claim_data.get('category', 'N/A')}")
                                with col_c:
                                    st.markdown(f"**Similarity:** {claim_data['similarity_score']:.2f}")
                    
                    # Technical Details
                    with st.expander("🔧 Technical Details"):
                        metadata = data["metadata"]
                        
                        col_tech1, col_tech2 = st.columns(2)
                        with col_tech1:
                            st.write(f"**Similar Claims Found:** {metadata.get('total_similar_claims', 0)}")
                            st.write(f"**Similarity Threshold:** {metadata.get('similarity_threshold', 'N/A')}")
                        
                        with col_tech2:
                            st.write(f"**Embedding Model:** {metadata.get('embedding_model', 'N/A')}")
                            st.write(f"**LLM Model:** {metadata.get('llm_model', 'N/A')}")
                
                else:
                    st.error(f"❌ Error: {result['error']}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>TrustAI v1.0.0 - Powered by OpenAI GPT-4 and Weaviate Vector Database</p>
        <p>🔗 <a href="http://localhost:8000/docs" target="_blank">API Documentation</a> | 
           🏥 <a href="http://localhost:8000/health" target="_blank">Health Check</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()