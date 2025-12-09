import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F # <-- Added F import

# --- Configuration (Ensure this matches your setup) ---
class PathsConfig:
    # Use the output path from your previous script
    labeled_data_parquet_path = "output/final_labeled_features.parquet"

# ---------------------------------------------------

def peek_at_parquet(parquet_path: str):
    """
    Reads the Parquet file, prints the schema, displays the first 5 rows, 
    and prints the count of unique TAC_Reading values.
    """
    print(f"--- Reading and peeking at: {parquet_path} ---")

    try:
        # 1. Initialize Spark Session (using local[*] for simplicity)
        spark = SparkSession.builder.master("local[*]").getOrCreate()
        
        # 2. Load the Parquet data
        df = spark.read.parquet(parquet_path)
        
        # 3. Print the Schema
        print("\n## 📋 DataFrame Schema")
        df.printSchema()

        # 4. Show the first few rows
        print("\n## 🔍 Sample Data (First 5 Rows)")
        df.show(5, truncate=False)
        
        # 5. Count the number of unique TAC_Reading values
        unique_tac_count = df.select(F.countDistinct("TAC_Reading")).collect()[0][0]
        
        print(f"\n## 🔢 Count of Truly Unique TAC_Reading Values")
        print(f"The total number of distinct TAC_Reading values is: {unique_tac_count}")

        # 6. Stop the Spark session
        spark.stop()
        
        # --- Interpretation of the count ---
        if unique_tac_count == 13:
            print("\n🚨 Interpretation: The distinct count is 13.")
            print("This confirms the hypothesis: each of the 13 PIDs was assigned a single, constant TAC value across all their accelerometer measurements.")
        
    except Exception as e:
        print(f"\nERROR: Could not read Parquet file or initialize Spark.")
        print(f"Details: {e}")


if __name__ == "__main__":
    peek_at_parquet(PathsConfig.labeled_data_parquet_path)