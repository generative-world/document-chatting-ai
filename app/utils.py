import functools

# In-memory cache for queries
cache = {}

def cached_query(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        query = args[0]
        if query in cache:
            return cache[query]
        else:
            result = func(*args, **kwargs)
            cache[query] = result
            return result
    return wrapper
