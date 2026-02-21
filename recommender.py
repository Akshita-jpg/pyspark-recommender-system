from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Create Spark Session
spark = SparkSession.builder \
    .appName("MovieRecommender") \
    .getOrCreate()

# Load dataset
ratings = spark.read.csv("data/ratings.csv", header=True, inferSchema=True)
ratings = ratings.select("userId", "movieId", "rating")

# Train-test split
(training, test) = ratings.randomSplit([0.8, 0.2])

# Build ALS Model
als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True,
    maxIter=10,
    regParam=0.1,
    rank=10
)

model = als.fit(training)

# Predictions
predictions = model.transform(test)

# Evaluation
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predictions)
print("RMSE:", rmse)

# Generate Recommendations
user_recs = model.recommendForAllUsers(5)
user_recs.show(5, truncate=False)

spark.stop()
