import numpy as np

from pendulum import datetime
from pyspark.sql import DataFrame

class FeatureExtractor:
    def __init__(self):
        pass

    def get_features(self, data: DataFrame) -> DataFrame:
        return data  # TODO implement actual feature extraction logic
    
    def _get_last_tac_given_time(self, data: DataFrame, pid: str, timestamp: int) -> float: # TODO maybe in preprocessing?
        """
        Get the last TAC reading before a given timestamp for a patient

        returns: float: the last TAC reading
        """

        pid_df = data[pid]
        
        closest_idx = np.argmax(pid_df['timestamp'] > timestamp)

        if closest_idx != 0: # adjust index iff not the first element
            closest_idx -= 1

        return pid_df.at[closest_idx, 'TAC_Reading'] # retrieves the TAC reading at the closest index
    

    def _is_intoxicated(self, tac_reading: float, threshold: float) -> bool: # TODO apply this as extra feature
        return tac_reading >= threshold
    

    # TODO maybe an is night feature?
    def _time_of_day_feature(self, timestamp: int) -> str:
        """
        Extract time of day feature from timestamp

        returns: str: 'morning', 'afternoon', 'evening', 'night'
        """
        hour = datetime.utcfromtimestamp(timestamp).hour

        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'
    
    

