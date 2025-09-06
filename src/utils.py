from pyspark.sql import SparkSession
# for enviornment configurations and path handling
import os
import sys

def get_spark(app_name="CropYeildPipeline"):
    """
    Create and return a Spark session.
    Ensures PySpark uses the same Python interpreter for driver and executors.
    """
    # python_path = sys.executable  # current Python interpreter
    # os.environ["PYSPARK_PYTHON"] = python_path
    # os.environ["PYSPARK_DRIVER_PYTHON"] = python_path


    #  Starts building the Spark session with the given application name.
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.local.dir","C:/spark-temp")
        .getOrCreate()
    )

    return spark
