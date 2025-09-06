# src/enrich_gold.py
# Purpose: Silver layer se data leke ML model ke liye features encode karna, vector banana, aur Gold layer mein save karna
import os
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from utils import get_spark

def main():
    spark = get_spark("CropYield-Gold")

    # Load Silver parquet
    silver_path = os.path.join("..","data", "silver", "crop_yield_v8")
    df = spark.read.parquet(silver_path)

    print("✅ Silver Data Loaded")
    df.printSchema()
    df.show(5, truncate=False)

    # -------------------
    # 1. Select Features + Target
    # -------------------
    # Adjust based on your dataset column names
    target_col = "Yield_tons_per_hectare"

    ## Dynamically select columns based on availability.
    ## your pipeline won't break if some columns are missing.
    categorical_cols = [c for c in ["Crop", "Region", "SoilType"] if c in df.columns]
    numeric_cols = [c for c in ["Rainfall_cm", "Temperature_C", "Rainfall_Deviation", "Soil_Fertility_Index"] if c in df.columns]

    stages = []

    # -------------------
    # 2. Encode Categorical Features
    # -------------------

    #Convert string categories into numeric vectors
    # - Why: ML algorithms need numerical input; this preserves category information without imposing ordinal relationships.

#     ➡️ Categorical columns ko pehle index kiya (string → number), fir one-hot encode kiya (number → vector).
#     ➡️ ML models ko numbers chahiye hote hain, strings nahi.

    for cat_col in categorical_cols:
        indexer = StringIndexer(inputCol=cat_col, outputCol=cat_col + "_idx", handleInvalid="keep")
        encoder = OneHotEncoder(inputCols=[cat_col + "_idx"], outputCols=[cat_col + "_vec"])
        stages += [indexer, encoder]

    # -------------------
    # 3. Assemble Features
    # -------------------

    #  Sab encoded categorical + numeric features ko ek single vector mein combine kiya — ye vector   ML model ka input hota hai.

    feature_cols = [c + "_vec" for c in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    stages.append(assembler)

    ## Chain all transformations into a reusable pipeline.
    ##  - Why: Keeps your workflow clean, modular, and reproducible

    from pyspark.ml import Pipeline
    pipeline = Pipeline(stages=stages)

    df_transformed = pipeline.fit(df).transform(df)

    # Final Gold dataset
    ## Extract the final feature vector and target column
    ## -  This gold_df is now ready for training a supervised ML model like logistic regression, decision trees, etc.
    gold_df = df_transformed.select(*categorical_cols,*numeric_cols,"features", F.col(target_col).alias("label"))

    print("✅ Gold Dataset Ready")
    gold_df.show(5, truncate=False)

    # -------------------
    # 4. Save Gold Layer
    # -------------------
    gold_path = os.path.join("..","data", "gold", "crop_yield_v8")
    gold_df.write.mode("overwrite").parquet(gold_path)

    print(f"✅ Gold data saved at: {gold_path}")
    spark.stop()

if __name__ == "__main__":
    main()