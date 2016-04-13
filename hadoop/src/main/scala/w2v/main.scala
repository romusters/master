/**
  * Created by robert on 13-4-16.
  */

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}

package hadoop {
	object w2v {

		def main(args: Array[String]): Unit = {

			val conf = new SparkConf().setAppName("w2v")
			val sc = new SparkContext(conf)
			val input = sc.textFile("/user/rmusters/20151225to31.tar").map(line => line.split(" ").toSeq)

			val word2vec = new Word2Vec()

			val model = word2vec.fit(input)

			val synonyms = model.findSynonyms("weer", 3)

			for((synonym, cosineSimilarity) <- synonyms) {
			  println(s"$synonym $cosineSimilarity")
			}

			// Save and load model
			model.save(sc, "/user/rmusters/2015122500.bin")
			val sameModel = Word2VecModel.load(sc, "/user/rmusters/20151225to31.bin")
		}
	}
}
