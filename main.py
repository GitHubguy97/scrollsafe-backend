from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time
import json
from datetime import datetime
from dotenv import load_dotenv
from video_utils import get_video_info
from heuristics import check_heuristics
import asyncio
from concurrent.futures import ThreadPoolExecutor
from huggingface_client import analyze_with_huggingface

# Load environment variables
load_dotenv()

app = FastAPI(title="ScrollSafe Backend", version="1.0.0")

# Create thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=4)  # Handle 4 concurrent requests

# Mock Doom Scroller Cache (simulates pre-analyzed viral content)
DOOM_SCROLLER_CACHE = {
    # AI-generated videos
    "ZiuUF14tJtY": { 
        "result": "ai-detected", 
        "confidence": 0.87, 
        "source": "DS Cache",
        "reason": "Deepfake detection model flagged this video"
    },   
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your extension's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisResult(BaseModel):
  result: str
  confidence: float
  reason: str
  source: str

# Routes
@app.get("/")
async def root():
    return {"message": "ScrollSafe Backend API"}

@app.get("/api/ds-cache/{video_id}")
async def check_doom_scroller_cache(video_id: str):
    """Check if video is in Doom Scroller pre-analyzed cache"""
    if video_id in DOOM_SCROLLER_CACHE:
        print(f"⚡ DS Cache hit for: {video_id}")
        cached = DOOM_SCROLLER_CACHE[video_id]
        return AnalysisResult(
            result=cached["result"],
            confidence=cached["confidence"],
            reason=cached["reason"],
            source=cached["source"]
        )
    else:
        # Return 404 for cache miss
        print(f"❌ DS Cache miss for: {video_id}")
        raise HTTPException(status_code=404, detail="Not in cache")

@app.get("/api/analyze/{video_id}")
async def analyze_video(video_id: str, request: Request):
    """Quick analysis: C2PA + metadata heuristics"""
    
    request_start = time.time()
    
    if await request.is_disconnected():
        print(f"⏭️ Client disconnected for {video_id}, aborting")
        raise HTTPException(status_code=499, detail="Client disconnected")
    
    # Get video metadata
    loop = asyncio.get_event_loop()
    video_info = await loop.run_in_executor(executor, get_video_info, video_id)

     # Check again after slow operation
    if await request.is_disconnected():
        print(f"⏭️ Client disconnected during processing for {video_id}")
        raise HTTPException(status_code=499, detail="Client disconnected")
    
    if not video_info:
        return AnalysisResult(
            result="unknown",
            confidence=0.0,
            reason="Could not fetch video information",
            source="Backend API"
        )
    
    # Run heuristics check
    heuristics_result = check_heuristics(video_info)
    
    # Calculate total time
    total_duration = time.time() - request_start
    
    # Log the complete request
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "video_id": video_id,
        "total_duration": round(total_duration, 3),
        "result": heuristics_result["result"],
        "confidence": heuristics_result["confidence"],
        "reason": heuristics_result["reason"]
    }
    
    with open("api_requests.log", 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    print(f"✅ Total request: {total_duration:.3f}s - Result: {heuristics_result['result']}")
    
    return AnalysisResult(
        result=heuristics_result["result"],
        confidence=heuristics_result["confidence"],
        reason=heuristics_result["reason"],
        source="Backend API"
    )

@app.get("/api/deep-scan/{video_id}")
async def deep_scan_video(video_id: str, request: Request):
    request_start = time.time()
    
    print(f"🤖 Starting deep scan for: {video_id}")
    
    if await request.is_disconnected():
        return
    
    # Run Hugging Face analysis
    result = await analyze_with_huggingface(video_id)
    
    total_duration = time.time() - request_start
    print(f"✅ Deep scan complete: {total_duration:.3f}s - Result: {result['result']}")
    
    return AnalysisResult(
        result=result["result"],
        confidence=result["confidence"],
        reason=result["reason"],
        source="Hugging Face ML Model"
    )

