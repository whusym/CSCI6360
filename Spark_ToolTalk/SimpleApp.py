"""SimpleApp.py"""
from pyspark.sql import SparkSession

logFile = "HMP_Dataset/README.txt"  # Should be some file on your system
spark = SparkSession.builder.appName("SimpleApp.py").master("local").getOrCreate()
logData = spark.read.text(logFile).cache()

numAs = logData.filter(logData.value.contains('a')).count()
numBs = logData.filter(logData.value.contains('b')).count()

print("Lines with a: %i, lines with b: %i" % (numAs, numBs))

spark.stop()
