import os
import time
import re
from typing import List, Dict, Any, Optional
import weaviate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from utils import calculate_trust_score, validate_claim_input, preprocess_text
from datetime import datetime

# Import web search fallback
try:
    from duckduckgo_search import DDGS
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    print("Warning: duckduckgo-search not installed. Fallback search disabled.")

load_dotenv()

class TrustAIRAGPipeline:
    def __init__(self):
        """Initialize the RAG pipeline with all necessary components."""
        self.weaviate_url = os.getenv("WEAVIATE_URL", "localhost")
        self.weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        self.max_retrieval_results = int(os.getenv("MAX_RETRIEVAL_RESULTS", "3"))
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_api_key=self.openai_api_key
        )
        
        self.llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4"),
            openai_api_key=self.openai_api_key,
            temperature=0.1
        )
        
        self.client = self._initialize_weaviate_client()
        self.fact_check_chain = self._create_fact_check_chain()
        self.general_analysis_chain = self._create_general_analysis_chain()
        self.web_search_analysis_chain = self._create_web_search_analysis_chain()
        self.research_synthesis_chain = self._create_research_synthesis_chain()
    
    def _initialize_weaviate_client(self):
        """Initialize Weaviate v3 client."""
        try:
            if self.weaviate_api_key and self.weaviate_api_key != "":
                auth_config = weaviate.AuthApiKey(api_key=self.weaviate_api_key)
                client = weaviate.Client(
                    url=f"https://{self.weaviate_url}",
                    auth_client_secret=auth_config
                )
            else:
                client = weaviate.Client(url="http://localhost:8080")
            
            # Test connection
            client.schema.get()
            print("Weaviate client connected successfully")
            return client
        except Exception as e:
            print(f"Failed to initialize Weaviate client: {e}")
            raise
    
    def _is_current_events_claim(self, claim: str) -> bool:
        """Determine if a claim needs current/recent information."""
    # Time indicators
        time_keywords = [
        'current', 'currently', 'now', 'today', 'recent', 'latest', 'this year',
        '2024', '2025', 'breaking', 'new', 'just announced'
        ]
    
    # Topics that change frequently
        dynamic_topics = [
        # Politics & Government
        'president', 'prime minister', 'election', 'government', 'congress',
        'biden', 'trump', 'harris', 'administration', 'policy',
        
        # Business & Economy  
        'ceo', 'stock price', 'company', 'merger', 'earnings', 'market',
        
        # Sports
        'plays for', 'team', 'traded', 'signed', 'coach', 'season',
        
        # Technology
        'released', 'launched', 'update', 'version', 'available',
        
        # Health/Medical (for ongoing research)
        'treatment', 'vaccine', 'study shows', 'research finds',
        
        # Entertainment
        'movie', 'show', 'album', 'tour', 'dating', 'married'
        ]
    
        claim_lower = claim.lower()
    
    # Check for explicit time indicators
        if any(keyword in claim_lower for keyword in time_keywords):
            return True
        
    # Check for dynamic topics
        if any(topic in claim_lower for topic in dynamic_topics):
            return True
        
        return False
    
    def _smart_web_research(self, claim: str) -> List[Dict[str, str]]:
        """Conduct intelligent web research using SerpAPI (Google Search)."""
        
        # Import SerpAPI
        try:
            from serpapi import GoogleSearch
            if not self.serpapi_key:
                print("Warning: SERPAPI_KEY not found. Falling back to DuckDuckGo")
                return self._fallback_search(claim)
        except ImportError:
            print("Warning: google-search-results not installed. Install with: pip install google-search-results")
            return self._fallback_search(claim)
        
        try:
            # Generate strategic search queries
            search_strategies = self._generate_search_strategies(claim)
            
            all_results = []
            seen_urls = set()
            
            for strategy in search_strategies[:3]:  # Try top 3 strategies
                try:
                    print(f"SerpAPI searching: {strategy['query']}")
                    
                    # Use SerpAPI for Google search
                    search = GoogleSearch({
                        "q": strategy['query'],
                        "api_key": self.serpapi_key,
                        "num": 6,  # Get 6 results per query
                        "gl": "us",  # Search from US
                        "hl": "en"   # English results
                    })
                    
                    results = search.get_dict()
                    
                    if "organic_results" in results:
                        for result in results["organic_results"]:
                            url = result.get('link', '')
                            if url in seen_urls:
                                continue
                            seen_urls.add(url)
                            
                            # Score source reliability
                            reliability_score = self._score_source_reliability_serpapi(result)
                            
                            if reliability_score > 0.3:  # Only include decent sources
                                all_results.append({
                                    'title': result.get('title', ''),
                                    'body': result.get('snippet', ''),
                                    'href': url,
                                    'snippet': result.get('snippet', '')[:400] + '...',
                                    'reliability_score': reliability_score,
                                    'search_strategy': strategy['type'],
                                    'position': result.get('position', 0)
                                })
                        
                        if len(all_results) >= 8:  # Enough results
                            break
                            
                except Exception as e:
                    print(f"SerpAPI search failed for {strategy['query']}: {e}")
                    continue
            
            # Sort by reliability score first, then by search position
            all_results.sort(key=lambda x: (x.get('reliability_score', 0), -x.get('position', 99)), reverse=True)
            
            return all_results[:6]  # Return top 6 results
            
        except Exception as e:
            print(f"SerpAPI research failed: {e}")
            return self._fallback_search(claim)
    
    def _generate_search_strategies(self, claim: str) -> List[Dict[str, str]]:
        """Generate universal search strategies for any topic like a professional fact-checking system."""
        strategies = []
    
    # Strategy 1: Primary Fact-Checking Sources (PolitiFact, Snopes, FactCheck.org)
        strategies.append({
        'query': f'"{claim}" site:politifact.com OR site:snopes.com OR site:factcheck.org',
        'type': 'primary_fact_checkers'
        })
    
    # Strategy 2: Wikipedia + Encyclopedia Sources
        strategies.append({
        'query': f'"{claim}" site:wikipedia.org OR site:britannica.com',
        'type': 'encyclopedia_sources'
        })
    
    # Strategy 3: Academic and Scientific Sources
        strategies.append({
        'query': f'"{claim}" site:pubmed.ncbi.nlm.nih.gov OR site:scholar.google.com OR site:nature.com',
        'type': 'academic_sources'
        })
    
    # Strategy 4: Major News and Verification
        strategies.append({
        'query': f'"{claim}" site:reuters.com OR site:ap.org OR site:bbc.com verification',
        'type': 'news_verification'
        })
    
    # Strategy 5: General Fact-Check Query
        strategies.append({
        'query': f'{claim} fact check verification truth debunked myth',
        'type': 'general_fact_check'
        })
    
    # Strategy 6: Current Information (if time-sensitive)
        if self._is_current_events_claim(claim):
            strategies.append({
            'query': f'{claim} 2024 2025 current latest',
            'type': 'current_information'
        })
    
    # Strategy 7: Government/Official Sources (for certain topics)
        strategies.append({
        'query': f'"{claim}" site:gov OR site:who.int OR site:cdc.gov',
        'type': 'government_official'
        })
    
        return strategies
    
    def _score_source_reliability_serpapi(self, result: Dict) -> float:
        """Score the reliability of a SerpAPI search result."""
        url = result.get('link', '').lower()
        title = result.get('title', '').lower()
        snippet = result.get('snippet', '').lower()
        
        score = 0.5  # Base score
        
        # TIER 1: Government and Official Sources (Highest Trust)
        tier1_domains = [
            '.gov', 'whitehouse.gov', 'congress.gov', 'senate.gov', 'house.gov',
            'cdc.gov', 'nih.gov', 'fda.gov', 'nasa.gov', 'state.gov',
            'who.int', 'un.org', 'europa.eu', 'parliament.uk'
        ]
        
        # TIER 2: Major News & Fact-Checkers (High Trust)  
        tier2_domains = [
            'reuters.com', 'ap.org', 'bbc.com', 'npr.org',
            'factcheck.org', 'snopes.com', 'politifact.com',
            'nytimes.com', 'washingtonpost.com', 'wsj.com',
            'cnn.com', 'abcnews.go.com', 'cbsnews.com', 'nbcnews.com'
        ]
        
        # TIER 3: Academic & Research (High Trust)
        tier3_domains = [
            'nature.com', 'science.org', 'nejm.org', 'jama.org',
            'plos.org', 'pubmed.ncbi.nlm.nih.gov', 'scholar.google.com',
            '.edu', 'mit.edu', 'harvard.edu', 'stanford.edu'
        ]
        
        # TIER 4: Encyclopedia & Reference (Medium-High Trust)
        tier4_domains = [
            'wikipedia.org', 'britannica.com', 'merriam-webster.com'
        ]
        
        # TIER 5: Other News (Medium Trust)
        tier5_domains = [
            'politico.com', 'thehill.com', 'guardian.com', 'economist.com',
            'time.com', 'newsweek.com', 'usnews.com'
        ]
        
        # LOW TRUST: Social Media, Blogs, Partisan Sources
        low_trust = [
            'facebook.com', 'twitter.com', 'reddit.com', 'youtube.com',
            'blog', 'wordpress', 'medium.com', 'quora.com',
            'donaldtrump.com', 'joebiden.com', 'truthsocial.com'
        ]
        
        # Score based on domain tier
        if any(domain in url for domain in tier1_domains):
            score += 0.45  # Government = highest trust
        elif any(domain in url for domain in tier2_domains):
            score += 0.35  # Major news/fact-checkers
        elif any(domain in url for domain in tier3_domains):
            score += 0.35  # Academic sources
        elif any(domain in url for domain in tier4_domains):
            score += 0.25  # Reference sources
        elif any(domain in url for domain in tier5_domains):
            score += 0.15  # Other news
        elif any(domain in url for domain in low_trust):
            score -= 0.4   # Penalize unreliable sources
        
        # Boost for fact-checking indicators
        fact_check_terms = [
            'fact check', 'fact-check', 'verify', 'verification', 'debunk', 
            'myth', 'false claim', 'true or false', 'reality check'
        ]
        if any(term in title or term in snippet for term in fact_check_terms):
            score += 0.2
        
        # Boost for official/authoritative language
        authoritative_terms = [
            'official', 'statement', 'press release', 'announcement',
            'confirmed', 'verified', 'according to', 'study shows'
        ]
        if any(term in title or term in snippet for term in authoritative_terms):
            score += 0.1
        
        # Boost for recent content (2024-2025)
        if '2025' in snippet or '2024' in snippet:
            score += 0.1
        
        # Penalty for opinion/editorial content
        opinion_terms = ['opinion', 'editorial', 'op-ed', 'blog', 'comment']
        if any(term in title or term in url for term in opinion_terms):
            score -= 0.2
        
        return max(0, min(1, score))
    
    def _fallback_search(self, claim: str) -> List[Dict[str, str]]:
        """Fallback to DuckDuckGo if SerpAPI fails."""
        if not WEB_SEARCH_AVAILABLE:
            return []
        
        try:
            with DDGS() as ddgs:
                results = []
                for result in ddgs.text(f"{claim} fact check", max_results=3):
                    results.append({
                        'title': result.get('title', ''),
                        'body': result.get('body', ''),
                        'href': result.get('href', ''),
                        'snippet': result.get('body', '')[:300] + '...',
                        'reliability_score': 0.5,
                        'search_strategy': 'fallback'
                    })
                return results
        except Exception as e:
            print(f"Fallback search failed: {e}")
            return []
    
    def _create_research_synthesis_chain(self) -> LLMChain:
        """Create chain for synthesizing multiple web sources like Claude."""
        prompt_template = PromptTemplate(
            input_variables=["user_claim", "web_sources", "current_date"],
            template="""You are a professional fact-checking expert analyzing a claim using multiple web sources.

CLAIM: {user_claim}
DATE: {current_date}

WEB RESEARCH RESULTS:
{web_sources}

Based on the web research above, provide a comprehensive fact-check analysis. Consider:

1. SOURCE QUALITY: Prioritize government sources (.gov), major news outlets (Reuters, AP, BBC), fact-checkers (PolitiFact, Snopes), and academic sources (.edu)
2. CONSENSUS: Look for agreement across multiple reliable sources
3. RECENCY: Favor more recent information, especially for current events
4. CONTEXT: Consider any important nuances or caveats

Provide your assessment in this format:
TRUST_SCORE: [0-100 where 0=completely false, 100=completely true]
CONFIDENCE: [0-100 based on source quality and consensus]
EXPLANATION: [Clear explanation of your reasoning, citing the most reliable sources]

Be especially careful with:
- Political claims (require official government sources)
- Health claims (require medical/scientific sources)
- Current events (require recent, reliable news sources)"""
        )
        return LLMChain(llm=self.llm, prompt=prompt_template)
    
    def _create_fact_check_chain(self) -> LLMChain:
        """Create LangChain chain for fact-checking with similar claims."""
        prompt_template = PromptTemplate(
            input_variables=["user_claim", "similar_claims"],
            template="""Analyze this claim for truthfulness: {user_claim}
            
Similar verified claims from our database: {similar_claims}

Based on the similar claims and your knowledge, provide a detailed analysis of the claim's accuracy."""
        )
        return LLMChain(llm=self.llm, prompt=prompt_template)
    
    def _create_general_analysis_chain(self) -> LLMChain:
        """Create LangChain chain for general fact-checking without similar claims."""
        prompt_template = PromptTemplate(
            input_variables=["user_claim"],
            template="""Analyze this claim for truthfulness: {user_claim}

Provide:
- A trust score from 0-100 (0 = completely false, 100 = completely true)
- A confidence level from 0-100 for your assessment
- A clear explanation of why you rated it this way

Format your response as:
TRUST_SCORE: [0-100]
CONFIDENCE: [0-100]
EXPLANATION: [detailed explanation]"""
        )
        return LLMChain(llm=self.llm, prompt=prompt_template)
    
    def _create_web_search_analysis_chain(self) -> LLMChain:
        """Create LangChain chain for analysis with web search results."""
        prompt_template = PromptTemplate(
            input_variables=["user_claim", "web_results", "current_date"],
            template="""Analyze this claim using web search results: {user_claim}

Current date: {current_date}
Web search results: {web_results}

Based on the web results, provide:
TRUST_SCORE: [0-100]
CONFIDENCE: [0-100]
EXPLANATION: [detailed explanation based on web results]"""
        )
        return LLMChain(llm=self.llm, prompt=prompt_template)
    
    def _parse_llm_assessment(self, llm_response: str) -> Dict[str, Any]:
        """Parse the LLM response to extract trust score, confidence, and explanation."""
        try:
            trust_score = 50
            confidence = 50
            explanation = llm_response
            
            # Extract trust score
            trust_match = re.search(r'TRUST_SCORE:\s*(\d+)', llm_response)
            if trust_match:
                trust_score = int(trust_match.group(1))
            
            # Extract confidence
            conf_match = re.search(r'CONFIDENCE:\s*(\d+)', llm_response)
            if conf_match:
                confidence = int(conf_match.group(1))
            
            # Extract explanation
            exp_match = re.search(r'EXPLANATION:\s*(.*)', llm_response, re.DOTALL)
            if exp_match:
                explanation = exp_match.group(1).strip()
            
            return {
                "trust_score": max(0, min(100, trust_score)),
                "confidence": max(0, min(100, confidence)),
                "explanation": explanation
            }
        except:
            return {
                "trust_score": 50,
                "confidence": 30,
                "explanation": llm_response
            }
    
    def fact_check_claim(self, claim: str) -> Dict[str, Any]:
        """Main method to fact-check a claim using the RAG pipeline."""
        start_time = time.time()
        
        try:
            validation = validate_claim_input(claim)
            if not validation["valid"]:
                return {
                    "trust_score": 0, "confidence": 0,
                    "explanation": validation["error"],
                    "processing_time": 0, "error": validation["error"]
                }
            
            # Check if this needs web research
            needs_web_search = self._is_current_events_claim(claim)
            
            # Generate embedding for database search
            processed_claim = preprocess_text(claim)
            embedding = self.embeddings.embed_query(processed_claim)
            
            # Search similar claims in database
            result = self.client.query.get("FactCheck", [
                "claim", "verdict", "explanation", "source", "confidence_score"
            ]).with_near_vector({
                "vector": embedding
            }).with_limit(self.max_retrieval_results).with_additional(["distance"]).do()
            
            similar_claims = []
            if "data" in result and "Get" in result["data"] and "FactCheck" in result["data"]["Get"]:
                for obj in result["data"]["Get"]["FactCheck"]:
                    distance = obj["_additional"]["distance"]
                    similarity = max(0, 1 - distance)
                    
                    if similarity >= self.similarity_threshold:
                        claim_data = {
                            "claim": obj.get("claim", ""),
                            "verdict": obj.get("verdict", ""),
                            "explanation": obj.get("explanation", ""),
                            "source": obj.get("source", ""),
                            "confidence_score": obj.get("confidence_score", 0),
                            "similarity_score": similarity
                        }
                        similar_claims.append(claim_data)
            
            # Decide analysis approach
            llm_analysis = ""
            trust_result = None
            analysis_type = "general_analysis"
            
            if needs_web_search:
                # Conduct smart web research
                print(f"Conducting web research for: {claim}")
                web_results = self._smart_web_research(claim)
                
                if web_results:
                    # Format results for LLM analysis
                    web_text = "\n\n".join([
                        f"Source {i+1} (Reliability: {r['reliability_score']:.2f}):\n"
                        f"Title: {r['title']}\n"
                        f"Content: {r['snippet']}\n"
                        f"URL: {r['href']}\n"
                        f"Search Type: {r['search_strategy']}"
                        for i, r in enumerate(web_results)
                    ])
                    
                    current_date = datetime.now().strftime("%B %d, %Y")
                    llm_analysis = self.research_synthesis_chain.run(
                        user_claim=claim,
                        web_sources=web_text,
                        current_date=current_date
                    )
                    trust_result = self._parse_llm_assessment(llm_analysis)
                    analysis_type = "web_research_serpapi"
                else:
                    # Fallback to general analysis
                    llm_analysis = self.general_analysis_chain.run(user_claim=claim)
                    trust_result = self._parse_llm_assessment(llm_analysis)
                    
            elif similar_claims:
                # Use database similar claims
                similar_text = "\n".join([f"- {c['claim']} (Verdict: {c['verdict']})" for c in similar_claims[:2]])
                llm_analysis = self.fact_check_chain.run(user_claim=claim, similar_claims=similar_text)
                
                similarity_scores = [c["similarity_score"] for c in similar_claims]
                trust_result = calculate_trust_score(similarity_scores, similar_claims)
                trust_result["explanation"] = f"{trust_result['explanation']}\n\nDetailed Analysis: {llm_analysis}"
                analysis_type = "similar_claims"
                
            else:
                # General analysis
                llm_analysis = self.general_analysis_chain.run(user_claim=claim)
                trust_result = self._parse_llm_assessment(llm_analysis)
                
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "trust_score": trust_result["trust_score"],
                "confidence": trust_result["confidence"],
                "explanation": trust_result["explanation"],
                "similar_claims": similar_claims,
                "llm_analysis": llm_analysis,
                "processing_time": round(processing_time, 2),
                "metadata": {
                    "total_similar_claims": len(similar_claims),
                    "similarity_threshold": self.similarity_threshold,
                    "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                    "llm_model": os.getenv("LLM_MODEL", "gpt-4"),
                    "analysis_type": analysis_type,
                    "web_search_used": needs_web_search,
                    "serpapi_enabled": bool(self.serpapi_key)
                }
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return {
                "trust_score": 0, "confidence": 0,
                "explanation": f"Error processing claim: {str(e)}",
                "processing_time": round(processing_time, 2),
                "error": str(e)
            }
    
    def close(self):
        """Close the Weaviate client connection."""
        pass