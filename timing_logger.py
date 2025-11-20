import time
import json
from datetime import datetime
from functools import wraps

def log_timing(filename="performance.log"):
    """Decorator to time function execution and log to file"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "function": func.__name__,
                "duration_seconds": round(duration, 3),
                "args": str(args)[:100],  # Truncate long args
            }
            
            # Append to log file
            with open(filename, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            print(f"⏱️  {func.__name__}: {duration:.3f}s")
            return result
        return wrapper
    return decorator

