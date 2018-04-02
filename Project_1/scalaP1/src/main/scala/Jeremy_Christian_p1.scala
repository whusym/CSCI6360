
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yuanming Shi and Christian McDaniel
 *  @version 1.0
 *  @date    Feb 5, 2018
 *  @see     LICENSE (MIT style license file)
 */

import scala.collection.immutable.ListMap
import scala.math

import scalation.linalgebra._
import scalation.plot.Plot
import scalation.random.CDF.studentTCDF
import scalation.util.{banner, Error, time}
import scalation.util.Unicode.sub
import scalation.stat.StatVector.corr
import scalation.util.getFromURL_File
import scalation.util.Error
import scalation.analytics.Regression
import scalation.analytics.SimpleRegression

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The 'RegressionTest' object uses the MatrixD and Regression classes 
* to perform multiple regression and subsequent analysis on a numerical dataset, read in as an 
* argument at the time of running the program.  
*  > sbt "run <filepath>"
*/
object RegressionTest extends App
{

    val data_path = args(0) // take in filepath as argument

    // set up your training and test datasets and variables
    val df = MatrixD(data_path, 1)   //skip the first row!
    val rows = df.dim1  // number of rows
    // val inter = VectoD(1)(df.dim1)
    val columns = df.dim2 //number of columns
    val df_x = df.sliceExclude(rows, 15)  // delete the last column (because it's Y)
    val df_y = df.col(15) //select the last column of original df as Y
    val df_i = df_y.copy  // copy the Y vector into another vector (for calculating the intercept)
    df_i.set(1) // set all values in the the new copied vector into 1
    val df_x1 = df_x.+^:(df_i) //concat the x matrix with the new vector

    // run your initial regression 
    val rg = new Regression(df_x1, df_y) //set a new regression
    rg.train()
    rg.report()
    println (rg.vif)

    // perform a transformation on X4 and rerun regression
    val df_x4sq = df_x1.col(4) * df_x1.col(4) // get x_4 ^2
    val df_trans = df_x1.:^+(df_x4sq)  // add the vector as a new column
    val rg_new = new Regression(df_trans, df_y) //set a new regression
    rg_new.train()
    rg_new.report()
    println (rg_new.vif)

    // drop cols suspected of not adding much informatin to data and rerun regression
    val df_dropCols = df_trans.sliceExclude(rows, 11).sliceExclude(rows, 5).sliceExclude(rows, 1) 
    val rg_dropped = new Regression(df_dropCols, df_y)
    rg_dropped.train()
    rg_dropped.report()
    println (rg_dropped.vif)
}