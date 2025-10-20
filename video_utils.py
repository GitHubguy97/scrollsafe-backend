from dotenv import load_dotenv
import requests
import os
import time
import re
from typing import Optional, Dict
from timing_logger import log_timing

load_dotenv()
api_key = os.getenv('YOUTUBE_API_KEY')

@log_timing()
def get_video_info(video_id: str) -> Optional[Dict]:
    """Get video metadata using YouTube Data API v3"""
    
    
    if not api_key:
        print("Error: YOUTUBE_API_KEY environment variable not set")
        return None
    
    try:
        # Time the API call
        api_start = time.time()
        
        # Call YouTube Data API v3
        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            'key': api_key,
            'id': video_id,
            'part': 'snippet,contentDetails,statistics'
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        api_duration = time.time() - api_start
        print(f"  └─ YouTube API call: {api_duration:.3f}s")
        
        if not data.get('items'):
            print(f"No video found for ID: {video_id}")
            return None
            
        video_data = data['items'][0]
        snippet = video_data['snippet']
        content_details = video_data.get('contentDetails', {})
        statistics = video_data.get('statistics', {})
        
        # Parse duration (ISO 8601 format like PT1M30S)
        duration_str = content_details.get('duration', 'PT0S')
        duration = parse_duration(duration_str)
        
        return {
            'video_id': video_id,
            'title': snippet.get('title'),
            'description': snippet.get('description', ''),
            'tags': snippet.get('tags', []),
            'uploader': snippet.get('channelTitle'),
            'upload_date': snippet.get('publishedAt', '').split('T')[0],  # YYYY-MM-DD
            'duration': duration,
            'view_count': int(statistics.get('viewCount', 0)),
            'like_count': int(statistics.get('likeCount', 0)),
            'width': None,  # Not available in API
            'height': None,  # Not available in API
        }
    except Exception as e:
        print(f"Error fetching video info: {e}")
        return None

def parse_duration(duration_str: str) -> int:
    """Parse ISO 8601 duration to seconds"""
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
    if not match:
        return 0
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    
    return hours * 3600 + minutes * 60 + seconds


# def download_video_chunk(video_id: str, byte_range: str = "0-50000") -> Optional[bytes]:
#     """Download first chunk of video for metadata analysis"""
#     url = f"https://www.youtube.com/shorts/{video_id}"
    
#     # Get video URL using yt-dlp
#     ydl_opts = {'quiet': True, 'format': 'best'}
    
#     try:
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             info = ydl.extract_info(url, download=False)
#             video_url = info['url']
            
#             # Download first chunk
#             headers = {'Range': f'bytes={byte_range}'}
#             response = requests.get(video_url, headers=headers)
            
#             return response.content
#     except Exception as e:
#         print(f"Error downloading video chunk: {e}")
#         return None