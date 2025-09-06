import os
from utils import get_spark
from pyspark.sql.functions import lit
from pyspark.sql.functions import rand

def main():
    spark = get_spark("CropYield-Bronze")
    spark.sparkContext.setLogLevel("ERROR")

    raw_path = os.path.join("..","data", "raw", "crop_yield.csv")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"CSV file not found at {raw_path}")
    
    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(raw_path)
    )

    print(f"Loaded Raw Data | Rows: {df.count()} | Columns: {len(df.columns)}")
    df.printSchema()
    df.show(10, truncate=False)

    # df = df.withColumn("AverageRainfall", lit(500.0))     # Example default value
    # df = df.withColumn("Nitrogen", lit(40.0))              # Soil nutrient placeholder
    # df = df.withColumn("Phosphorus", lit(25.0))
    # df = df.withColumn("Potassium", lit(35.0))

    df = df.withColumn("AverageRainfall", lit(500.0))     # Example default value
    df = df.withColumn("Nitrogen", (rand()*50))              # Soil nutrient placeholder
    df = df.withColumn("Phosphorus", (rand()*30))
    df = df.withColumn("Potassium", (rand()*40))

    bronze_path = os.path.join("..","data", "bronze", "crop_yield_v8")
    df.repartition(1).write.mode("overwrite").parquet(bronze_path)
    print(f"Bronze data written to {bronze_path}")
    spark.stop()

if __name__ == "__main__":
    main()