import scalation.random.Normal
import scalation.linalgebra.{VectorD, VectorI, VectoD}
import scalation.plot.Plot
import scala.math.log
import java.io._

class Classifier (x: VectorD, xx: Array [VectorD], y: VectorI, z: VectorD)
{
    //if (z == null) {z = VectorD(x)}

    var term1: VectorD = null
    var term2: VectorD = null
    var mu:    VectorD = null
    var py:    VectorD = null
    var delta: VectorD = null
    var sig2:  Double = 0.94

    val m = 200
    val md = m.toDouble

    def train(xx: Array [VectorD]): Classifier =
    {
      mu = VectorD (xx.map(_.mean))
      mu.toDouble
      py = VectorD (xx.map(_.dim/md))

      var sum = 0.0
      for (c <- 0 until 2) sum += (xx(c) - mu(c)).normSq
      sig2  = sum / (m - 2).toDouble

      term1 = mu/sig2
      term2 = mu~^2 / (2.0 * sig2) - py.map (log (_))

      //println (term1)
      //println (term2)

      this
    } // train

    def classify (z: Double): Int =
    {
      val delta = term1 * z - term2
      val c = delta.argmax ()
      return c
    } // classify

    def using[T <: Closeable, R](resource: T)(block: T => R): R = {
      try { block(resource) }
      finally { resource.close() }
    } // using

} // Classifier

object ClassifierTest extends App
{
    val normal1 = Normal(98.6, 1.0)
    val normal2 = Normal(101, 1.0)

    var x = new VectorD(200)
    var y = new VectorI(200)
    var z: VectorD = null


    for (i <- 0 until 100 ) {
      x(i) = normal1.gen
      y(i) = 0
    } // for re normal1

    for (i <- 100 until 200) {
      x(i) = normal2.gen
      y(i) = 1
    } // for re normal2


    val xx = Array(VectorD(for (i <- x.range if y(i) == 0) yield x(i)), VectorD(for (i <- x.range if y(i) == 1) yield x(i)))

    val cl = new Classifier(x, xx, y, z)
    cl.train(xx)

    var yp = new VectorI(200)
    for (i <- 0 until 200) {
      yp(i) = cl.classify(x(i))
    } // for classify

    val t = VectorD.range(0,200)
    val plot1 = new Plot(t,y.toDouble, yp.toDouble)
    val plot2 = new Plot(x, y.toDouble, yp.toDouble)

    var tp, tn, fn, fp = 0
    for (i <- 0 until 200) {
        if (y(i) + yp(i) == 2) tp += 1
        if (y(i) + yp(i) == 0) tn += 1
        if (y(i) + yp(i) == 1 && y(i) == 1) fn += 1
        if (y(i) + yp(i) == 1 && y(i) == 0) fp += 1
    } // for accuracy

    println(tp)
    println(tn)
    println(fp)
    println(fn)

    cl.using(new BufferedWriter(new OutputStreamWriter(new FileOutputStream("~/Documents/CSCI6360/Project_2/CSCI6360_P2_Data3_Flu.xlsx")))) {
      writer =>
        for (i <- 0 until 200) {
          writer.write(x(i) + "\t" + y(i) + "\n")
        } // for writer
    } // using()

} // ClassifierTest
