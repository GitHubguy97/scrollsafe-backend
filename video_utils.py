import yt_dlp
import requests
import time
from typing import Optional, Dict
from timing_logger import log_timing

@log_timing()
def get_video_info(video_id: str) -> Optional[Dict]:
    """Get video metadata using yt-dlp without downloading"""
    url = f"https://www.youtube.com/shorts/{video_id}"
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
    try:
        # Time the yt-dlp extraction specifically
        ydl_start = time.time()
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        
        ydl_duration = time.time() - ydl_start
        print(f"  └─ yt-dlp extraction: {ydl_duration:.3f}s")
        
        
        return {
            'video_id': video_id,
            'title': info.get('title'),
            'uploader': info.get('uploader'),
            'duration': info.get('duration'),
            'upload_date': info.get('upload_date'),
            'description': info.get('description', ''),
            'tags': info.get('tags', []),
            'format': info.get('format'),
            'fps': info.get('fps'),
            'width': info.get('width'),
            'height': info.get('height'),
        }
    except Exception as e:
        print(f"Error fetching video info: {e}")
        return None


def download_video_chunk(video_id: str, byte_range: str = "0-50000") -> Optional[bytes]:
    """Download first chunk of video for metadata analysis"""
    url = f"https://www.youtube.com/shorts/{video_id}"
    
    # Get video URL using yt-dlp
    ydl_opts = {'quiet': True, 'format': 'best'}
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_url = info['url']
            
            # Download first chunk
            headers = {'Range': f'bytes={byte_range}'}
            response = requests.get(video_url, headers=headers)
            
            return response.content
    except Exception as e:
        print(f"Error downloading video chunk: {e}")
        return None