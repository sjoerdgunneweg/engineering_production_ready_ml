from datetime import datetime

def unix_timestamp_to_readable(unix_timestamp: int) -> str:
    """Convert a Unix timestamp to a human-readable string format.

    Args:
        unix_timestamp (int): The Unix timestamp to convert.    
    Returns:
        str: The human-readable string representation of the timestamp.
    """

    readable_time = datetime.utcfromtimestamp(unix_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return readable_time