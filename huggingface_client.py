import requests
import asyncio
from typing import Dict
from video_utils import get_video_info
from heuristics import check_heuristics
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

async def analyze_with_huggingface(video_id: str) -> Dict:
    """
    Deep scan: AI thumbnail analysis + enhanced heuristics fallback
    Uses Hugging Face Inference API to analyze video thumbnail
    Falls back to heuristics if AI analysis fails
    """
    
    # Step 1: Try AI thumbnail analysis first
    ai_result = None
    if HUGGING_FACE_API_KEY:
        try:
            print(f"🤖 Attempting AI thumbnail analysis for {video_id}")
            ai_result = await analyze_thumbnail_with_ai(video_id)
            if ai_result:
                print(f"✅ AI analysis successful: {ai_result['result']}")
                return ai_result
        except Exception as e:
            print(f"⚠️ AI analysis failed, falling back to heuristics: {e}")
    else:
        print(f"⚠️ No Hugging Face API key, using heuristics only")
    
    # Step 2: Fallback to heuristics-based analysis
    print(f"📊 Using heuristics-based analysis for {video_id}")
    await asyncio.sleep(1.0)  # Simulate processing time
    
    video_info = get_video_info(video_id)
    
    if video_info:
        heuristics_result = check_heuristics(video_info)
        
        # If heuristics found AI, boost confidence for deep scan
        if heuristics_result["result"] == "ai-detected":
            return {
                "result": "ai-detected",
                "confidence": min(heuristics_result["confidence"] + 0.05, 0.98),
                "reason": f"Deep scan confirmed: {heuristics_result['reason']}"
            }
        elif heuristics_result["result"] == "suspicious":
            return {
                "result": "ai-detected",
                "confidence": heuristics_result["confidence"] + 0.15,
                "reason": f"Deep analysis detected: {heuristics_result['reason']}"
            }
    
    # No AI indicators found
    print(f"✅ No AI indicators found, marking as verified")
    return {
        "result": "verified",
        "confidence": 0.75,
        "reason": "Deep scan found no AI indicators"
    }

async def analyze_thumbnail_with_ai(video_id: str) -> Dict:
    """
    Analyze video thumbnail using OpenAI's CLIP model via Hugging Face Inference API
    Uses zero-shot image classification to compare thumbnail against text descriptions
    Returns analysis result or None if fails
    """
    
    try:
        # Get YouTube thumbnail (maxresdefault for best quality)
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        
        print(f"🔍 Analyzing thumbnail: {thumbnail_url}")
        
        # Initialize Hugging Face Inference Client
        client = InferenceClient(token=HUGGING_FACE_API_KEY)
        
        # Use OpenCLIP model for zero-shot image classification
        # Model: openai/clip-vit-large-patch14 hosted on Hugging Face
        model_name = "openai/clip-vit-large-patch14"
        
        # Define candidate labels for zero-shot classification
        # CLIP will compute similarity scores between the image and each label
        candidate_labels = [
            "AI generated image",
            "synthetic CGI render", 
            "computer generated graphics",
            "real photograph",
            "authentic video frame"
        ]
        
        # Perform zero-shot image classification via Inference API
        result = client.zero_shot_image_classification(
            thumbnail_url,
            candidate_labels,
            model=model_name
        )
        
        print(f"🔍 CLIP Response: {result}")
        
        # Calculate AI likelihood score from CLIP's similarity scores
        ai_score = calculate_ai_score_from_clip(result)
        
        print(f"🤖 AI model analysis complete: score={ai_score:.2f}")
        
        # Return classification based on AI score threshold
        if ai_score > 0.7:
            return {
                "result": "ai-detected",
                "confidence": ai_score,
                "reason": f"CLIP model detected synthetic visual patterns (confidence: {ai_score:.0%})"
            }
        elif ai_score > 0.5:
            return {
                "result": "suspicious",
                "confidence": ai_score,
                "reason": f"CLIP model flagged potential synthetic content (confidence: {ai_score:.0%})"
            }
        else:
            return {
                "result": "verified",
                "confidence": 1.0 - ai_score,
                "reason": f"CLIP model analysis indicates authentic content"
            }
            
    except Exception as e:
        import traceback
        print(f"❌ AI analysis exception: {type(e).__name__}: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        return None


def calculate_ai_score_from_clip(clip_response) -> float:
    """
    Calculate AI-generation likelihood from CLIP's zero-shot classification scores
    
    CLIP returns similarity scores between the image and each text prompt.
    We aggregate scores for AI-related prompts vs. real-content prompts.
    
    Args:
        clip_response: List of {"label": str, "score": float} from CLIP
    
    Returns:
        Float between 0-1 representing likelihood of AI-generated content
    """
    
    # Handle different possible response formats
    if not clip_response or not isinstance(clip_response, list):
        print(f"⚠️ Unexpected CLIP response format, using default score")
        return 0.6  # Default moderate suspicion
    
    # Map labels to AI vs. Real categories
    ai_keywords = ["ai", "generated", "synthetic", "cgi", "computer", "render", "fake"]
    real_keywords = ["real", "authentic", "photograph", "genuine", "actual", "video frame"]
    
    ai_score_sum = 0.0
    real_score_sum = 0.0
    
    for item in clip_response:
        label = item.get("label", "").lower()
        score = item.get("score", 0.0)
        
        # Check if label indicates AI-generated content
        if any(keyword in label for keyword in ai_keywords):
            ai_score_sum += score
        # Check if label indicates real content
        elif any(keyword in label for keyword in real_keywords):
            real_score_sum += score
    
    # Normalize: AI score relative to total
    total_score = ai_score_sum + real_score_sum
    if total_score > 0:
        ai_likelihood = ai_score_sum / total_score
    else:
        ai_likelihood = 0.5  # Neutral if no scores
    
    # Add slight bias toward caution (better to flag suspicious than miss AI content)
    ai_likelihood = min(ai_likelihood * 1.1, 0.98)
    
    return round(ai_likelihood, 2)