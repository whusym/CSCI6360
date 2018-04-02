//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Christian McDaniel
 *  @version 1.4
 *  @date    14 Mar 2018
 *  @see     LICENSE (MIT style license file).
 */

package scalation

import scala.math.exp
import scala.util.control.Breaks.{break, breakable}

import scalation.linalgebra.{MatriD, MatrixD, VectoD, VectorD}
import scalation.math.FunctionS2S
import scalation.calculus.Differential.FunctionV_2V
import scalation.random.RandomVecD
import scalation.util.{banner, Error}

import scalation.analytics.ActivationFunc._

class Perceptron2 (x: MatriD, y: VectoD, private var eta: Double = 1.0,
                  afunc: FunctionS2S = sigmoid _, afuncV: FunctionV_2V = sigmoidV _)
	//extends Predictor with Error
{

	val wg = RandomVecD(x.dim2)
	val b = wg.gen
	var sse = Double.MaxValue
	var sseo = sse 
	private val MAX_ITER = 200   
	val _1 = VectorD.one(y.dim)
	val EPSILON = 1E-4

	def min_err	(b: VectorD)
	{

		// val t = x dot b 

		for (it <- 0 until MAX_ITER)
		{
			val yp = afuncV (x dot b)
			val e = y - yp
			// b is based on the derivative of the activation function 
			b += x.t * (e * yp * (_1 - yp)) * eta	// this needs to change for different activation functions 
			sseo = sse
			sse = e dot e 
			if ( (sseo - sse) < EPSILON )
				return 

		} // for 

	} // min_err