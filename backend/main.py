from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from rag_pipeline import TrustAIRAGPipeline

load_dotenv()

pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Starting TrustAI application...")
    try:
        pipeline = TrustAIRAGPipeline()
        print("RAG pipeline initialized successfully")
    except Exception as e:
        print(f"Failed to initialize RAG pipeline: {e}")
        raise
    yield
    print("Shutting down TrustAI application...")
    if pipeline:
        pipeline.close()

app = FastAPI(
    title=os.getenv("APP_NAME", "TrustAI"),
    version=os.getenv("APP_VERSION", "1.0.0"),
    description="Real-time misinformation detection system using RAG pipeline",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClaimRequest(BaseModel):
    claim: str = Field(..., min_length=10, max_length=1000)
    include_similar_claims: bool = Field(default=True)
    include_llm_analysis: bool = Field(default=True)

class ClaimResponse(BaseModel):
    trust_score: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=100)
    explanation: str
    similar_claims: Optional[List[Dict[str, Any]]] = None
    llm_analysis: Optional[str] = None
    processing_time: float
    metadata: Dict[str, Any]

start_time = time.time()

def get_pipeline():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return pipeline

@app.get("/")
async def root():
    return {
        "message": "Welcome to TrustAI - Misinformation Detection API",
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/check_claim", response_model=ClaimResponse)
async def check_claim(
    request: ClaimRequest,
    rag_pipeline: TrustAIRAGPipeline = Depends(get_pipeline)
):
    try:
        result = rag_pipeline.fact_check_claim(request.claim)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        response_data = {
            "trust_score": result["trust_score"],
            "confidence": result["confidence"],
            "explanation": result["explanation"],
            "processing_time": result["processing_time"],
            "metadata": result["metadata"]
        }
        
        if request.include_similar_claims:
            response_data["similar_claims"] = result["similar_claims"]
        
        if request.include_llm_analysis:
            response_data["llm_analysis"] = result["llm_analysis"]
        
        return ClaimResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        uptime = time.time() - start_time
        pipeline_ready = pipeline is not None
        
        status = "healthy" if pipeline_ready else "unhealthy"
        
        return {
            "status": status,
            "pipeline_ready": pipeline_ready,
            "uptime": round(uptime, 2)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "pipeline_ready": False,
            "uptime": round(time.time() - start_time, 2),
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    print(f"Starting TrustAI API server on {host}:{port}")
    print(f"Debug mode: {debug}")
    print(f"API docs available at: http://{host}:{port}/docs")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )
