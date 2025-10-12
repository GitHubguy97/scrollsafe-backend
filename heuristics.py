import re
from typing import Dict, Optional
from timing_logger import log_timing

@log_timing()
def check_heuristics(video_info: Dict) -> Dict:
    """Check video metadata for suspicious patterns"""
    
    if not video_info:
        return {
            "result": "unknown",
            "confidence": 0.0,
            "reason": "Could not fetch video metadata"
        }
    
    # Combine text to check
    text_to_check = (
        video_info.get('title', '') + ' ' + 
        video_info.get('description', '') + ' ' +
        ' '.join(video_info.get('tags', []))
    ).lower()
    
    # High confidence AI patterns - definitive AI detection
    high_confidence_patterns = [
        # Explicit AI generation/creation phrases
        (r'\bai[-\s](generated|created|animated|produced|made|powered|content|video|baby|animation)', 
         0.92, "Explicitly labeled as AI-generated"),
        
        # Common AI tools
        (r'\b(midjourney|stable[-\s]?diffusion|dall[-\s]?e|runway|synthesia)', 
         0.90, "AI tool mentioned"),
        
        # Deepfake explicitly mentioned
        (r'\bdeep\s*fake\b', 0.95, "Deepfake explicitly mentioned"),
        
        # Ultra-realistic + AI combo
        (r'\bultra[-\s]?realistic\b.*\bai\b|\bai\b.*\bultra[-\s]?realistic\b', 
         0.88, "Ultra-realistic AI content indicator"),
        
        # AI-related hashtags (moved from medium)
        (r'#ai(generated|created|animated|baby|content|video)', 0.85, "AI generation hashtag found"),
        
        # Channel names with AI indicators
        (r'@\w*ai[-_](creator|animation|content|generated)', 0.82, "AI-focused creator channel"),
        
        # "Artificial intelligence" spelled out (moved from medium)
        (r'\bartificial\s+intelligence\b', 0.80, "Artificial intelligence mentioned"),
        
        # Synthetic content
        (r'\bsynthetic\b', 0.78, "Synthetic content mentioned"),
    ]
    
    # Check high confidence patterns first
    for pattern, confidence, reason in high_confidence_patterns:
        match = re.search(pattern, text_to_check, re.IGNORECASE)
        if match:
            matched_text = match.group(0)
            return {
                "result": "ai-detected",
                "confidence": confidence,
                "reason": f"{reason}: '{matched_text}'"
            }
    
    # Medium confidence patterns - suspicious but not definitive
    medium_confidence_patterns = [
        # General AI-related hashtags
        (r'#ai\w*', 0.68, "AI-related hashtag found"),
        
        # Standalone "AI" word (less aggressive now)
        (r'\bai\b', 0.60, "AI mentioned in content"),
        
        # Common AI-related terms
        (r'\b(neural|machine\s*learning)\b', 
         0.55, "AI-related technical terms"),
    ]
    
    # Check medium confidence patterns
    for pattern, confidence, reason in medium_confidence_patterns:
        match = re.search(pattern, text_to_check, re.IGNORECASE)
        if match:
            matched_text = match.group(0)
            return {
                "result": "suspicious",
                "confidence": confidence,
                "reason": f"{reason}: '{matched_text}'"
            }
    
    # Check for unusual aspect ratios (AI often uses 1:1)
    width = video_info.get('width', 0)
    height = video_info.get('height', 0)
    
    if width and height:
        aspect_ratio = width / height
        if abs(aspect_ratio - 1.0) < 0.1:  # Close to 1:1
            return {
                "result": "suspicious",
                "confidence": 0.50,
                "reason": "Unusual 1:1 aspect ratio common in AI-generated content"
            }
    
    # Check for very short duration
    duration = video_info.get('duration', 0)
    if duration and duration < 5:
        return {
            "result": "suspicious",
            "confidence": 0.45,
            "reason": "Very short duration may indicate AI-generated clip"
        }
    
    # No suspicious patterns found
    return {
        "result": "unknown",
        "confidence": 0.0,
        "reason": "No suspicious heuristics detected"
    }