
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Christian McDaniel and Jeremy Shi
 *  @version 1.4
 *  @date    14 Mar 2018
 *  @see     LICENSE (MIT style license file).
 */

package scalation

import scalation.analytics.classifier.{NullModel, NaiveBayes, NaiveBayes0, TANBayes, TANBayes0, ClassifierInt, LDA, LogisticRegression, KNN_Classifier}
import scalation.columnar_db.Relation
import scalation.random.PermutedVecI
import scalation.random.RNGStream.ranStream
import scalation.linalgebra._
import scala.math.sqrt


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ClassifierTests` class reads in three datasets of varying sizes and 
*   complexities, each with binomial target values, and  performs six different 
*	classification algorithms (NullModel, NaiveBayes, TANBayes, LDA, Logistic
*	Regression and KNN) from the ScalaTion library. 
*
*	The data is first partitioned into the specified number of folds for k-fold
*	cross validation. We have used 10-fold and 20-fold CV on each dataset, with
*	the exception of using 14-fold instead of 20-fold CV for the Tennis dataset
*	since it only contains 14 examples (i.e., effectively leave-one-out CV). 
*
*	For each fold, the true positives (tp), true negatives (tn), false positives 
*	(fp) and false negatives (fn) are measured and used to calculate performance 
*	measures including accuracy, recall, precision and F Score. Beta value of 1
*	is used for F-score so as to weight the precision and recall equally, since
*	the datasets come from various sources and may not share the same optimal 
*	weighting. 
*
*	To avoid dividing by zero when calculating precision, recall and F-Score, 
*	especially likely when the dataset is small (e.g., the Tennis dataset 
*	included in this class), we followed advice found at the link below,
*	which instructs to set precision, recall and F-score to 1 if tp, fp and fn
*	are all 0; otherwise, set precision, recall and F-score to 0 if tp is 0 
*	and the other two are greater than 0. 
*
*	The code can be run as follows: 
*
*		1) Move the project folder to inside the scalation_1.4 library. You will 
*		   likely need to copy the .jar files from the /target folder of 
*		   scalation_modeling into the /lib fol der of this class. 
*		2) Ensure the caravan.csv and breat-cancer.arff datasets have been added 
*		   to the folder "data/analytics/classifier" within the main scalation
*		   library. Otherwise, change the BASE_DIR variable or otherwise load in
*		   your own data. 
*		3) Designate which datasets and folds you would like to run by leaving
*		   the call to the main method, crossValidateAlgos(), with those 
*		   parameters uncommented at the bottom of this script, or otherwise
*		   write your own call to the crossValidateAlgos() method. Comment
*		   out those you do not wish to run. Save the changes. 
*		4) From your command line, navigate into the ClassifierTests folder and
*		   enter the following command:
*		   
*				$ sbt run 
*
*		   NOTE: Due to high verbosity in the DEBUG and other output features of 
*		   some of the classifiers, the output for successive calls to the 
*		   crossValidateAlgos() method may be hard to find, so setting this DEBUG 
*		   feature to false and commenting out some of this output in the source
*		   files, and then repackaging the entire scalation_modeling package may
*		   be helpful for having concise output for multiple cross validations. 
*		   Be sure to move the new .jar files from the /target folder in 
*		   scalation_modeling to the /lib folder of this class. Otherwise, consider
*		   only running one cross validation at a time. 
*
*	@see https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
*	
*	Sample output has been provided within the "output" folder of this class. 
*			
*/


object ClassifierTests extends App
{
    private val BASE_DIR = "../data/analytics/classifier/" // cancer and caravan data saved within data/analytics/classifier folder of scalation library 										

    // load the ExampleTennis dataset
    import scalation.analytics.classifier.ExampleTennis._
    val xyTennis    = xy 																				// data already build as a MatriI matrix 
    val xTennis     = xyTennis                  (0 until xyTennis.dim1, 0 until xyTennis.dim2 - 1)		// separate data 
    val yTennis     = xyTennis.col              (xyTennis.dim2 - 1)										// from labels
    val fnTennis    = fn 																				// feature names
    val cnTennis    = cn 																				// class names

    // load and prepare the Caravan dataset
    val caravanFP   = BASE_DIR                  + "caravan.csv" 
    val dataCaravan = Relation                  (caravanFP, "Caravan", -1, null, ",") 					// read in the CSV
    val xyCaravan   = dataCaravan.toMatriI2     (null) 													// convert to MatriI datatype 
    val xCaravan    = xyCaravan                 (0 until xyCaravan.dim1, 0 until xyCaravan.dim2 - 1)	// separate data
    val yCaravan    = xyCaravan.col             (xyCaravan.dim2 - 1)									// from labels
    val fnCaravan   = dataCaravan.colName.slice (0, xyCaravan.dim2-1).toArray 							// feature names
    val cnCaravan   = Array                     ("No", "Yes")											// class names

    // load and prepare the breast-cancer.arff data
    val brstCncrDta = BASE_DIR                  + "breast-cancer.arff" 	
    var dataCancer  = Relation                  (brstCncrDta, -1, null) 								// read in arff file
    val xyCancer    = dataCancer.toMatriI2      (null) 													// convert to MatriI datatype
    val xCancer     = xyCancer                  (0 until xyCancer.dim1, 0 until xyCancer.dim2 - 1)		// separate data
    val yCancer     = xyCancer.col              (xyCancer.dim2 - 1)										// from labels
    val fnCancer    = dataCancer.colName.slice  (0, xyCancer.dim2 - 1).toArray 							// feature names
    val cnCancer    = Array                     ("p", "e")                        						// class names

    val k = 2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute and return accuracy, recall, precision and F-Score from the 
    *	provided statistics of the current fold, which are provided by the 
    *	checkClass() method below. 
    *	@param tp  the true  positives from current classifier during current fold
    *	@param tn  the true  negatives from current classifier during current fold
    *	@param fp  the false positives from current classifier during current fold
    *	@param fn  the false negatives from current classifier during current fold
    */
    def getStats (tp :Double, tn :Double, fp :Double, fn :Double): (Double, Double, Double, Double) =
    {
      var foldAcc    = (tp + tn) / (tp + tn + fp + fn) 													
      var foldRecall = 0.0
      var foldPrecis = 0.0
      var foldFScore = 0.0

      if (tp == 0.0) {												// set recall, precision and FScore 
          if ((fp == 0.0) & (fn == 0.0)) {							// to predesignated value if tp == 0
              foldRecall = 1.0
              foldPrecis = 1.0
              foldFScore = 1.0
          } else {
              foldRecall = 0.0
              foldPrecis = 0.0
              foldFScore = 0.0
          }
      } else {
        foldRecall = tp / (tp + fn)
        foldPrecis = tp / (tp + fp)
        foldFScore = 2 * foldPrecis * foldRecall / (foldPrecis + foldRecall)
      } // else tp not 0

      (foldAcc, foldRecall, foldPrecis, foldFScore)
    } // getStats

	//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	/** Determine the confusion matrix parameters, namely tp, tn, fp, fn, for
	*	each example within the given fold's validation set for the current
	*	classifier. Feed these parameters to getStats() above and return the 
	*	results.
	*	@param predictArray  the predicted values for each validation example
	*	@param actualArray   the ground truth label for each validation example
	*	@param show          if true, prints some informative messages; 
	*						 false by default
	*/
    def checkClass (predictArray: VectorD, actualArray: VectorD, show: Boolean = false): (Double, Double, Double, Double) =
      {
        if (show) {
          println(s"predictions: $predictArray")
          println(s"class: $actualArray")
        } // if show

        var fp = 0.0
        var fn = 0.0
        var tp = 0.0
        var tn = 0.0

        for (pred <- 0 until predictArray.dim)  {

          if (predictArray(pred) == 0.0) {
              if (actualArray(pred) == 0)      tn += 1.0
              else if (actualArray(pred) == 1) fn += 1.0
          } // if predict neg

          else if (predictArray(pred) == 1.0) {
              if (actualArray(pred) == 1) tp += 1.0
              else if (actualArray(pred) == 0) fp += 1.0
          } // if predict pos

        } // for

        if (show) {
          println(s"tp: $tp, fp: $fp, tn: $tn, fn: $fn")
        } // if show

        var (acc, recall, precision, fScore) = getStats(tp, tn, fp, fn)	// call the getStats() method above
        (acc, recall, precision, fScore)
      } // checkClass


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The main method for the `ClassifierTests` class. Performs six different 
    *	classifiers on the provided data via k-fold cross validation. During 
    *	each fold, feeds the classifications for each classifier to the method
    *	checkClass() above to calculate performance measures, and then 
    *	collects these metrics for mean and standard deviation calculations. 
    *	Prints the mean and standard deviation for each classifier to the 
    *	console for review. 
    *	@param dataset    short name for identifying the dataset during output
    *	@param datasetxy  the training dataset with labels still attached
    *	@param datasetx   the unlabeled training data
    *	@param datasety   the ground trught labels for the training data
    *	@param datasetfn  the feature names for the dataset
    *	@param datasetcn  the class names for the dataset
    *	@param nx 		  the number of folds for k-fold cross validation
    *	@param caravan    if true, indicates that the caravan dataset is being 
    *					  used, for which stochastic gradient descent is used
    *					  for optimizing parameters during logistic regression,
    *					  as opposed to QuasiNewton methods for other datasets;
    *					  false by default
    *	@param show		  if true, prints some informative messages; 
	*					  false by default
    */
    def crossValidateAlgos (dataset :String, datasetxy :MatriI, datasetx :MatriI, datasety :VectoI, datasetfn :Array[String], datasetcn :Array[String], nx :Int, caravan: Boolean = false, show: Boolean = false)
        {

            val vc          = (for(j <- datasetx.range2) yield datasetx.col(j).max() + 1).toArray		// calculate the value counts for each feature 
            if (show) {for (i <- vc) println(s"value count at index $i: $vc(i)")}

            val permutedVec = PermutedVecI     (VectorI.range (0, datasetx.dim1), ranStream)
            val randOrder   = permutedVec.igen                       									// randomize integers 0 until size
            val itestA      = randOrder.split  (nx)                   									// make array of itest indices

            if (show) {for (i <- itestA) println(s"randomly generated list of indices for the fold-wise testing: $i")}

            var accNM       = new VectorD (0)															// declare the containers for fold-specific performance metrics
            var recallNM    = new VectorD (0)															// for each classifier
            var precisNM    = new VectorD (0)
            var fScoreNM    = new VectorD (0)

            var accNB       = new VectorD (0)
            var recallNB    = new VectorD (0)
            var precisNB    = new VectorD (0)
            var fScoreNB    = new VectorD (0)

            var accTAN      = new VectorD (0)
            var recallTAN   = new VectorD (0)
            var precisTAN   = new VectorD (0)
            var fScoreTAN   = new VectorD (0)

            var accLDA      = new VectorD (0)
            var recallLDA   = new VectorD (0)
            var precisLDA   = new VectorD (0)
            var fScoreLDA   = new VectorD (0)

            var accLR       = new VectorD (0)
            var recallLR    = new VectorD (0)
            var precisLR    = new VectorD (0)
            var fScoreLR    = new VectorD (0)

            var accKNN      = new VectorD (0)
            var recallKNN   = new VectorD (0)
            var precisKNN   = new VectorD (0)
            var fScoreKNN   = new VectorD (0)

            for (it <- 0 until nx) {                													// for loop for cross validation

                var classesNM  = new VectorD (0)														// declare the containers for the fold-specific classifications
                var classesNB  = new VectorD (0)														// for each classifier
                var classesTAN = new VectorD (0)
                var classesLDA = new VectorD (0)
                var classesLR  = new VectorD (0)
                var classesKNN = new VectorD (0)

                val itest      = itestA(it)() 															// get array from it element
                if (show) {println(s"randomly generated validation set for current fold: $itest")}      
                var rowsList   = Array[Int]() 															// the training row indices to be kept for training 
                for (rowi <- 0 until datasetx.dim1) {
                    if (!itest.contains(rowi)) {rowsList = rowsList :+ rowi} 							// build the fold-specific training set row indices
                } // for    
                if (show) println(s"rows to include for training: length = $rowsList.length")

                val foldx  = datasetx.selectRows(rowsList)												// build the fold-specific training set
                val foldxR = Relation.fromMatriI (foldx, "foldx", datasetfn)							// convert from MatriI (used by Null and Bayes models)
                val foldxI = foldxR.toMatriD (0 until foldx.dim2)										// to 
                val foldxD = new MatrixD (foldxI)														// MatrixD for LDA, logistic regression, KNN
                val foldy  = datasety.select(rowsList)													// build the fold-specific training set
                if (show) println(s"initial dataset has $datasetx.dim1 rows; fold training set has $foldx.dim1 rows")
                if (show) {println(s"MatrixI training set: $foldx\nMatrixD training set: $foldxD\nTraining labels: $foldy\n")}

                var nm  = new NullModel          (        foldy,            k, datasetcn)				// construct new classifiers 	
                var nb  = new NaiveBayes         (foldx,  foldy, datasetfn, k, datasetcn,    vc)		// providing only the training data/labels
                var tan = new TANBayes           (foldx,  foldy, datasetfn, k, datasetcn, 0, vc)
                var lda = new LDA                (foldxD, foldy, datasetfn,    datasetcn)
                var lr  = new LogisticRegression (foldxD, foldy, datasetfn)
                var knn = new KNN_Classifier     (foldxD, foldy, datasetfn, k, datasetcn)

                nm.train  ()								
                nb.train  ()
                tan.train ()                                    										// train 
                lda.train ()
                if (caravan) {lr.train_SGD ()} 															
                else {lr.train  ()}
                knn.train ()

                var rowy :Int = 10000																	// initialize ground truth label to high number for debugging
                var yArray    = new VectorD (0)															// initialize container for ground truth labels (classifier never sees these)

                   
                for (ic <- itest) {																		// for loop for classifying each example in the validation set

                    var rowx = datasetx (ic)															// validation example 
                    rowy     = datasety (ic)															// ground truth label 

                    if (show) println(s"row: $rowx")
                    
                    var (iclassNM,  icnNM,  iporbNM)  = nm.classify  ( rowx )							// classify exmaple for each classifier
                    var (iclassNB,  icnNB,  iprobNB)  = nb.classify  ( rowx )
                    var (iclassTAN, icnTAN, iprobTAN) = tan.classify ( rowx )
                    var (iclassLDA, icnLDA, iprobLDA) = lda.classify ( rowx )
                    var (iclassLR,  icnLR,  iprobLR)  = lr.classify  ( rowx )
                    var (iclassKNN, icnKNN, iprobKNN) = knn.classify ( rowx )

                    yArray     = yArray     ++ rowy														// append label/prediction to containers 
                    classesNM  = classesNM  ++ iclassNM
                    classesNB  = classesNB  ++ iclassNB
                    classesTAN = classesTAN ++ iclassTAN
                    classesLDA = classesLDA ++ iclassLDA
                    classesLR  = classesLR  ++ iclassLR
                    classesKNN = classesKNN ++ iclassKNN

                } // for classify

                // get fold-specific performance measures for each classifier
                var (foldAccNM,  foldRecallNM,  foldPrecisionNM,  foldFScoreNM)  = checkClass (classesNM,  yArray, show)	
                var (foldAccNB,  foldRecallNB,  foldPrecisionNB,  foldFScoreNB)  = checkClass (classesNB,  yArray, show)
                var (foldAccTAN, foldRecallTAN, foldPrecisionTAN, foldFScoreTAN) = checkClass (classesTAN, yArray, show)
                var (foldAccLDA, foldRecallLDA, foldPrecisionLDA, foldFScoreLDA) = checkClass (classesLDA, yArray, show)
                var (foldAccLR,  foldRecallLR,  foldPrecisionLR,  foldFScoreLR)  = checkClass (classesLR,  yArray, show)
                var (foldAccKNN, foldRecallKNN, foldPrecisionKNN, foldFScoreKNN) = checkClass (classesKNN, yArray, show)

                accNM     = accNM     ++ foldAccNM														// collect fold-wise performance measures
                recallNM  = recallNM  ++ foldRecallNM													// for each classifier
                precisNM  = precisNM  ++ foldPrecisionNM
                fScoreNM  = fScoreNM  ++ foldFScoreNM

                accNB     = accNB     ++ foldAccNB
                recallNB  = recallNB  ++ foldRecallNB
                precisNB  = precisNB  ++ foldPrecisionNB
                fScoreNB  = fScoreNB  ++ foldFScoreNB

                accTAN    = accTAN    ++ foldAccTAN
                recallTAN = recallTAN ++ foldRecallTAN
                precisTAN = precisTAN ++ foldPrecisionTAN
                fScoreTAN = fScoreTAN ++ foldFScoreTAN

                accLDA    = accLDA    ++ foldAccLDA
                recallLDA = recallLDA ++ foldRecallLDA
                precisLDA = precisLDA ++ foldPrecisionLDA
                fScoreLDA = fScoreLDA ++ foldFScoreLDA

                accLR     = accLR     ++ foldAccLR
                recallLR  = recallLR  ++ foldRecallLR
                precisLR  = precisLR  ++ foldPrecisionLR
                fScoreLR  = fScoreLR  ++ foldFScoreLR

                accKNN    = accKNN    ++ foldAccKNN
                recallKNN = recallKNN ++ foldRecallKNN
                precisKNN = precisKNN ++ foldPrecisionKNN
                fScoreKNN = fScoreKNN ++ foldFScoreKNN

            } // for cv

            // print the output (mean and std dev performance for each metric for each classifier)

            val formatHeader = "%10s %10s %10s %10s %10s\n"
            val formatMain   = "%-10s %10.9f %10.9f %10.9f %10.9f\n"

            val methodNM     = "NullModel"
            val methodNB     = "NaiveBayes"
            val methodTAN    = "TANBayes"
            val methodLDA    = "LDA"
            val methodLR     = "LR"
            val methodKNN    = "KNN"

            println ()
            println ("|====================|")
            println(s"|  $dataset DATASET  |")
            println(s"|     $nx   FOLDS     |")
            println ("|====================|")
            println ()

            println(">meanCV")
            printf(formatHeader, "", "accuracy", "recall", "precision", "f-score")
            printf(formatMain, methodNM,  accNM.mean,  recallNM.mean,  precisNM.mean,  fScoreNM.mean)
            printf(formatMain, methodNB,  accNB.mean,  recallNB.mean,  precisNB.mean,  fScoreNB.mean)
            printf(formatMain, methodTAN, accTAN.mean, recallTAN.mean, precisTAN.mean, fScoreTAN.mean)
            printf(formatMain, methodLDA, accLDA.mean, recallLDA.mean, precisLDA.mean, fScoreLDA.mean)
            printf(formatMain, methodLR,  accLR.mean,  recallLR.mean,  precisLR.mean,  fScoreLR.mean)
            printf(formatMain, methodKNN, accKNN.mean, recallKNN.mean, precisKNN.mean, fScoreKNN.mean)
            println("==========================================================")
            println()
            println(">stdCV")
            printf(formatHeader, "", "accuracy", "recall", "precision", "f-score")
            printf(formatMain, methodNM,  sqrt(accNM.variance),  sqrt(recallNM.variance),  sqrt(precisNM.variance),  sqrt(fScoreNM.variance))
            printf(formatMain, methodNB,  sqrt(accNB.variance),  sqrt(recallNB.variance),  sqrt(precisNB.variance),  sqrt(fScoreNB.variance))
            printf(formatMain, methodTAN, sqrt(accTAN.variance), sqrt(recallTAN.variance), sqrt(precisTAN.variance), sqrt(fScoreTAN.variance))
            printf(formatMain, methodLDA, sqrt(accLDA.variance), sqrt(recallLDA.variance), sqrt(precisLDA.variance), sqrt(fScoreLDA.variance))
            printf(formatMain, methodLR,  sqrt(accLR.variance),  sqrt(recallLR.variance),  sqrt(precisLR.variance),  sqrt(fScoreLR.variance))
            printf(formatMain, methodKNN, sqrt(accKNN.variance), sqrt(recallKNN.variance), sqrt(precisKNN.variance), sqrt(fScoreKNN.variance))
            println("==========================================================")
            println()


        } // crossValidateRand

  // call the methods to run the classifiers with the given data for given folds: 
  //10-fold CV
  crossValidateAlgos (" TENNIS ", xyTennis,  xTennis,  yTennis,  fnTennis,  cnTennis,  10)
  crossValidateAlgos ("CARAVAN ", xyCaravan, xCaravan, yCaravan, fnCaravan, cnCaravan, 10, true)
  crossValidateAlgos (" CANCER ", xyCancer,  xCancer,  yCancer,  fnCancer,  cnCancer,  10)
 
  //20-fold CV
  crossValidateAlgos (" TENNIS ", xyTennis,  xTennis,  yTennis,  fnTennis,  cnTennis,  14)              // LOOCV
  crossValidateAlgos ("CARAVAN ", xyCaravan, xCaravan, yCaravan, fnCaravan, cnCaravan, 20, true)
  crossValidateAlgos (" CANCER ", xyCancer,  xCancer,  yCancer,  fnCancer,  cnCancer,  20)

} // ClassifierTests
