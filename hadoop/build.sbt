lazy val root = (project in file(".")).
  settings(
    name := "w2v",
    version := "1.0",
    scalaVersion := "2.11.4",
    mainClass in Compile := Some("hadoop.w2v")        
  )

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.1" % "provided",
  "org.apache.spark" %% "spark-mllib" % "1.6.1" % "provided"
)

// META-INF discarding
mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
   {
    case PathList("META-INF", xs @ _*) => MergeStrategy.discard
    case x => MergeStrategy.first
   }
}



//libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "1.6.1"
//libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "1.6.1"
