ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.14"

lazy val root = (project in file("."))
  .settings(
    name := "library-dualnumbers",
    idePackagePrefix := Some("com.example.dualnumber"),
    libraryDependencies += "org.scalax" %% "scalax-chart" % "0.5.3"
  )
