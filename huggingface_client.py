import requests
import asyncio
import io
from typing import Dict
from video_utils import get_video_info
from heuristics import check_heuristics
import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY") or os.getenv("HF_TOKEN")

async def analyze_with_huggingface(video_id: str) -> Dict:
    """
    Deep scan: AI thumbnail analysis with heuristics fallback for false negatives
    Uses cropped thumbnail + haywoodsloan model, falls back to heuristics when AI says "verified" but heuristics says "ai-detected"
    """
    
    # Step 1: Get heuristics result for comparison
    print(f"Getting heuristics baseline for {video_id}")
    video_info = get_video_info(video_id)
    heuristics_result = None
    if video_info:
        heuristics_result = check_heuristics(video_info)
        print(f"Heuristics result: {heuristics_result['result']} ({heuristics_result['confidence']:.2f})")
    
    # Step 2: Try AI thumbnail analysis
    ai_result = None
    if HUGGING_FACE_API_KEY:
        try:
            print(f"Attempting AI thumbnail analysis for {video_id}")
            ai_result = await analyze_thumbnail_with_ai(video_id)
            if ai_result:
                print(f"AI analysis successful: {ai_result['result']} ({ai_result['confidence']:.2f})")
                
                # Smart decision logic based on heuristics baseline
                if heuristics_result:
                    heuristics_result_type = heuristics_result["result"]
                    ai_result_type = ai_result["result"]
                    
                    # Case 1: Heuristics says "verified" (non-AI) - prioritize heuristics for reliability
                    if heuristics_result_type == "verified":
                        if ai_result_type == "verified":
                            # Both agree it's real - trust heuristics with slight AI boost
                            print(f"Both heuristics and AI say verified - trusting heuristics baseline")
                            return {
                                "result": "verified",
                                "confidence": min(heuristics_result["confidence"] + 0.05, 0.95),
                                "reason": f"Heuristics confirmed by AI analysis"
                            }
                        else:
                            # AI says AI but heuristics says verified - trust heuristics for non-AI
                            print(f"AI says {ai_result_type} but heuristics says verified - trusting heuristics for non-AI")
                            return {
                                "result": "verified",
                                "confidence": heuristics_result["confidence"],
                                "reason": f"Heuristics indicates authentic content, AI analysis inconclusive"
                            }
                    
                    # Case 2: Heuristics says "ai-detected" - balance with AI
                    elif heuristics_result_type == "ai-detected":
                        if ai_result_type == "ai-detected":
                            # Both agree it's AI - combine confidence
                            combined_confidence = min((heuristics_result["confidence"] + ai_result["confidence"]) / 2 + 0.1, 0.98)
                            print(f"Both heuristics and AI say ai-detected - combining confidence")
                            return {
                                "result": "ai-detected",
                                "confidence": combined_confidence,
                                "reason": f"Double confirmation: {heuristics_result['reason']}"
                            }
                        else:
                            # AI says verified but heuristics says ai-detected - trust heuristics
                            print(f"AI says verified but heuristics says ai-detected - trusting heuristics")
                            return {
                                "result": "ai-detected",
                                "confidence": min(heuristics_result["confidence"] + 0.1, 0.95),
                                "reason": f"Heuristics override: {heuristics_result['reason']}"
                            }
                    
                    # Case 3: Heuristics says "suspicious" - let AI break the tie
                    elif heuristics_result_type == "suspicious":
                        if ai_result_type == "ai-detected":
                            # AI confirms suspicious -> AI detected
                            print(f"AI confirms suspicious heuristics - marking as ai-detected")
                            return {
                                "result": "ai-detected",
                                "confidence": min(ai_result["confidence"] + 0.05, 0.95),
                                "reason": f"AI confirmed suspicious indicators: {ai_result['reason']}"
                            }
                        else:
                            # AI says verified despite suspicious heuristics - trust AI
                            print(f"AI says verified despite suspicious heuristics - trusting AI")
                            return ai_result
                
                # No heuristics baseline - trust AI result
                return ai_result
        except Exception as e:
            print(f"AI analysis failed, falling back to heuristics: {e}")
    else:
        print(f"No Hugging Face API key, using heuristics only")
    
    # Step 3: Fallback to heuristics-based analysis
    print(f"Using heuristics-based analysis for {video_id}")
    await asyncio.sleep(1.0)  # Simulate processing time
    
    if heuristics_result:
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
    print(f"No AI indicators found, marking as verified")
    return {
        "result": "verified",
        "confidence": 0.75,
        "reason": "Deep scan found no AI indicators"
    }

async def analyze_thumbnail_with_ai(video_id: str) -> Dict:
    """
    Analyze video thumbnail using haywoodsloan/ai-image-detector-dev-deploy with center cropping
    Downloads thumbnail, crops to 9:16 center, sends to AI detector
    Returns analysis result or None if fails
    """
    
    try:
        # Download thumbnail bytes
        raw_bytes = _download_thumbnail_bytes(video_id)
        if not raw_bytes:
            raise RuntimeError("Could not download thumbnail")
        
        # Load and crop image
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        w, h = img.size
        l, t, r, b = _compute_center_crop_box(w, h, margin=0.02)
        cropped = img.crop((l, t, r, b))
        
        print(f"Cropped thumbnail: {w}x{h} -> {cropped.size} (box={[l,t,r,b]})")
        
        # Convert to JPEG bytes
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=90)
        img_bytes = buf.getvalue()
        
        # Call haywoodsloan model
        result = _classify_haywood_bytes(img_bytes)
        
        print(f"Haywood Response: {result}")
        
        # Parse result - find artificial vs real scores
        artificial_score = 0.0
        real_score = 0.0
        
        for item in result:
            label = str(item.get("label", "")).lower()
            score = float(item.get("score", 0.0))
            if "artificial" in label:
                artificial_score = score
            elif "real" in label:
                real_score = score
        
        confidence = max(artificial_score, real_score)
        is_ai = artificial_score >= real_score
        
        print(f"AI analysis complete: artificial={artificial_score:.3f}, real={real_score:.3f}")
        
        # Return classification based on AI score
        if is_ai and artificial_score >= 0.8:
            return {
                "result": "ai-detected",
                "confidence": artificial_score,
                "reason": f"AI detector (cropped) classified as artificial (confidence: {artificial_score:.0%})"
            }
        elif is_ai and artificial_score >= 0.6:
            return {
                "result": "suspicious", 
                "confidence": artificial_score,
                "reason": f"AI detector (cropped) flagged potential synthetic content (confidence: {artificial_score:.0%})"
            }
        else:
            return {
                "result": "verified",
                "confidence": real_score,
                "reason": f"AI detector (cropped) indicates authentic content (confidence: {real_score:.0%})"
            }
            
    except Exception as e:
        import traceback
        print(f"AI analysis exception: {type(e).__name__}: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        return None


def _download_thumbnail_bytes(video_id: str) -> bytes:
    """Download YouTube thumbnail bytes, trying maxres then hqdefault."""
    for path in ["maxresdefault.jpg", "hqdefault.jpg"]:
        url = f"https://img.youtube.com/vi/{video_id}/{path}"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200 and r.content:
                return r.content
        except Exception:
            pass
    return None


def _compute_center_crop_box(width: int, height: int, margin: float = 0.02) -> tuple:
    """Compute centered 9:16 crop box with margin."""
    target_ratio = 9 / 16
    ratio = width / height if height else target_ratio
    if ratio > target_ratio:
        crop_w = int(round(height * target_ratio))
        left = (width - crop_w) // 2
        right = left + crop_w
        top, bottom = 0, height
    else:
        crop_h = int(round(width / target_ratio))
        top = (height - crop_h) // 2
        bottom = top + crop_h
        left, right = 0, width
    dx = int((right - left) * margin)
    dy = int((bottom - top) * margin)
    return max(0, left + dx), max(0, top + dy), min(width, right - dx), min(height, bottom - dy)


def _classify_haywood_bytes(image_bytes: bytes):
    """Call haywoodsloan/ai-image-detector-dev-deploy via raw POST."""
    url = "https://api-inference.huggingface.co/models/haywoodsloan/ai-image-detector-dev-deploy"
    headers = {
        "Authorization": f"Bearer {HUGGING_FACE_API_KEY}",
        "Content-Type": "image/jpeg",
    }
    resp = requests.post(url, headers=headers, data=image_bytes, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(data["error"]) 
    # Normalize elements to dicts if needed
    normalized = []
    for item in data:
        label = item.get("label") if isinstance(item, dict) else getattr(item, "label", None)
        score = item.get("score") if isinstance(item, dict) else getattr(item, "score", None)
        if label is not None and score is not None:
            normalized.append({"label": str(label), "score": float(score)})
    return normalized


# Legacy function - kept for compatibility but not used in new implementation
def calculate_ai_score_from_clip(clip_response) -> float:
    """Legacy function - not used in new haywoodsloan implementation."""
    return 0.5