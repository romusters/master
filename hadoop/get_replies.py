appName = "replies"
sc = SparkContext(appName=appName)
from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)

data = sqlContext.read.json("data.json")
reply = data.select("in_reply_to_status_id")

reply = reply.where(reply.in_reply_to_status_id.isNotNull())

freqs = reply.groupBy(reply.in_reply_to_status_id).count()
freqs.write.format("com.databricks.spark.csv").mode('overwrite').save('reply_freqs.csv')

freqs.describe().show()


for row in x:
	new_row = []
	exp_row = np.exp(row)
	sum_row = exp_row.sum()
	exp_row = [el/sum_row for el in exp_row]
	new_x.append(exp_row)