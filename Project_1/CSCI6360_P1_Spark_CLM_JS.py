"""CSCI6360_P1_Spark_CLM_JS.py"""

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.sql import Column
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.mllib import linalg
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import ArrayType
from argparse import ArgumentParser
"""
__author__ = "Christian McDaniel and Jeremy Shi"

This python file reads in data and performs several data processing operations
as well as multiple linear regression and associated statistics computation.

"""

def data_to_frame(fp):
    """
    Reads in the data; constructs and prints a Spark Dataframe.

    Parameters:
    ___________________
    fp: (str) filepath of csv file conataining the data

    Returns:
    ___________________
    df: (Spark Dataframe) Dataframe containing data - rows = training examples;
        cols = features and response

    """
    # Read in the data
    p1_data = spark.read.format("csv").option("header", "true").option("inferSchema","true").load(fp)
    # create the dataframe
    expr = [col(c).cast("Double").alias(c) for c in p1_data.columns]
    df = p1_data.select(*expr)
    print('The DataFrame is:')
    df.show()
    return df

def mult_lin_reg(data, input_cols, response_col):
    """
    Fits a linear regression model for the data/repsonse vector provided by
    first creating a single vector containing the feature values for each
    training example using VectorAssembler(); next, constructs and fits a
    linear regression model using LinearRegression().

    Prints intercept and coefficients of model.

    Parameters:
    _________________
    data:         dataframe-style object created by data_to_frame()
    input_cols:   the columns of data param to be used as training features
    response_col: the header of the column in input data containing labels

    Returns:
    _________________
    lrModel:
    """
    # create the feature vectors for linear regression
    assembler = VectorAssembler(
        inputCols=input_cols,
        outputCol="features")
    df_lr = assembler.transform(data)
    print (df_lr.show())
    # run linear regression on the data; print stats
    lr = LinearRegression(featuresCol="features",\
                            labelCol = response_col)
    lrModel = lr.fit(df_lr)
    # Get all col names except Y (for calculating coefficients and other quality measures)
    col_names = list(data.schema.names)
    col_names.remove('Y')
    col_names.append('Intercept')
    print("\n", "Intercept: ", lrModel.intercept, "\n")
    print("Coefficients: ")
    print('\n'.join(list(str((feature, coeff)) for feature, coeff in zip(col_names, lrModel.coefficients))))
    return lrModel

def model_stats(data, lr_model, *args):
    """
    Calculates the summary statistic(s) indicated in *args and prints the
    results. Statistic names must come from pre-specified list of options
    (see below)

    Parameters:
    _________________
    lr_model: linear regression model returned by mult_lin_reg()
    *args:    variable number of string arguments representing the desired
              summary statistic. Options include: "R2", "CSE", "DoF", "MSE",
              "RSS","t-Values", "p-Values"
    """
    lrSummary = lr_model.summary
    last_row  = len(lrSummary.coefficientStandardErrors)
    # get the names of all feature columns and the intercept column
    col_names = list(data.schema.names)
    col_names.remove('Y')
    col_names.append('Intercept')

    if "R2" in args:
        print ("R2: ", lrSummary.r2)
    if "CSE" in args:
        print("\nCoefficient Std Err: ")
        print('\n'.join(list(str((feature, err)) for feature, err in \
                zip(col_names, lrSummary.coefficientStandardErrors))))
        print("**Row {} is the intercept**".format(last_row))
    if "DoF" in args:
        print("\nDoF: ", lrSummary.degreesOfFreedom)
    if "MSE" in args:
        print("\nMSE: ", lrSummary.meanSquaredError)
    if "RSS" in args or "SSE" in args:
        print("\nRSS/SSE: ", lrSummary.meanSquaredError * data.count())
    if "t-Values" in args:
        print("\nt-Values: ")
        print('\n'.join(list(str((feature, tval)) for feature, tval in \
                zip(col_names, lrSummary.tValues))))
        print("**Row {} is the intercept**".format(last_row))
    if "p-Values" in args:
        print("\n\np-Values: ")
        print('\n'.join(list(str((feature, pval)) for feature, pval in \
                zip(col_names, lrSummary.pValues))))
        print("**Row {} is the intercept**".format(last_row))
    print()
    return

def exp_transformation(data, orig_col, new_col, power):
    """
    Performs feature transformation by raising the indicated feature column to
    the given power.

    Parameters:
    _________________
    data:     dataframe containing training data
    orig_col: (str) column in data to be transformed
    new_col:  (str) header for new transformed column
    power:    (int) number to raise the original column by

    Returns:
    _________________
    transformed_df: new dataframe with transformed column added as the last col
    """
    transformed_df = data.withColumn(new_col, pow(data[orig_col], power))
    print('The transformed DataFrame is:')
    transformed_df.show()
    return transformed_df

def cols_to_drop(statistic, threshold, greater):
    """
    Compiles a list of columns to be dropped by drop_cols(), using the given
    summary statistics.

    Parameters:
    _________________
    statisic:        Statistic to use for feature evaluation; should have
                     <lr_model>.summary.<statistic>
    threshold:       (float) Number used as the threshold for feature evaluation
    greater:         (bool) If true, dop columns which are greater than the
                     given threshold; otherwise, drop columns which are less
                     than the given threshold

    Returns:
    _________________
    cols_list: (list) list of columns to be dropped by drop_cols()
    """
    cols_list = []
    if greater:
        for val in range(len(statistic)):
            if statistic[val] >= threshold:
                cols_list.append("X{}".format(val+1))
    else:
        for val in range(len(statistic)):
            if statistic[val] <= threshold:
                cols_list.append("X{}".format(val+1))
    return cols_list

def drop_cols(df, cols):
    """
    Removes from the input dataframe the columns indicated in args argument.
    Prints error message if columns can't be dropped.

    Parameters:
    _________________
    input_df: dataframe containing columns to be removed
    cols:     (list) columns to be removed

    Returns:
    _________________
    df_after_drop: dataframe with desired columns dropped, if no exception raised
    """
    try:
        for col in cols:
            df = df.drop(col)
            print('The DataFrame after dropping columns is:')
            df.show()
        return df
    except:
        print("Something went wrong... make sure column names provided are in dataframe")
        return

def vif(input_data, y_included):
    """
    Computes the Variance Inflation Factor (vif) for each feature in the data
    by iteratively regressing each feature against the rest of the data.

    Parameters:
    ______________________
    input_data: (Spark or other Dataframe) The input dataframe
    y_included: (bool) True indicates that the last column is the response vector

    Returns:
    ______________________
    vif_values: (dictionary) the feature name - vif pairing for each feature in the input data
    """
    vif_values = {}
    feature = 0
    if y_included:
        end = -1
    else:
        end = None

    while feature < len(input_data.columns[:end]):
        #drop the current feature from the dataframe
        df_minus_feature = input_data.drop(input_data[feature])
        # assemble the feature vector for MLR
        assembler_vif = VectorAssembler(
            inputCols=df_minus_feature.columns[:-1],
            outputCol="features_vif")
        input_data_lr = assembler_vif.transform(input_data)
        # regress the dropped column against the rest of the variables
        lr_vif = LinearRegression(featuresCol='features_vif',\
                        labelCol = input_data.columns[feature])
        lrModel_vif = lr_vif.fit(input_data_lr)
        lrSummary_vif = lrModel_vif.summary
        # add the R2 value to the dictionary
        vif_values[input_data.columns[feature]] = 1.0 / (1.0-lrSummary_vif.r2)
        feature += 1
    print("VIF: {}".format(vif_values))
    return vif_values

if __name__ == '__main__':
    # initiate the Spark session
    spark = SparkSession.builder.getOrCreate()

    # add an argument parser for the dataset filepath
    parser = ArgumentParser()
    parser.add_argument(dest='filepath', help = 'specify the path of the dataset')
    args = parser.parse_args()
    filepath = args.filepath

    # set up variables
    lr1_input_features = ["X" + str(x+1) for x in range(15)]
    lr2_input_features = lr1_input_features + ["X4^2"]
    lr_label           = 'Y'

    # read in data and create initial dataframe
    orig_df = data_to_frame(filepath)

    # perform linear regression on original dataframe
    orig_lr_model = mult_lin_reg(orig_df, lr1_input_features, lr_label)
    model_stats(orig_df, orig_lr_model, "R2", "CSE", "DoF", "MSE", "RSS","t-Values", "p-Values")

    # perform transformation on one column of data and fit linear model
    df_with_X4sq = exp_transformation(orig_df, "X4", "X4^2", 2)
    transformed_lr_model = mult_lin_reg(df_with_X4sq, lr2_input_features, lr_label)
    model_stats(df_with_X4sq, transformed_lr_model, "R2")

    # drop columns based on some statistic and fit linear model
    list_of_cols = cols_to_drop(transformed_lr_model.summary.pValues, 0.1, True)
    df_dropped_cols = drop_cols(df_with_X4sq, list_of_cols)

    lr3_input_features = df_dropped_cols.columns[:-2]
    lr3_input_features.append(df_dropped_cols.columns[-1])
    dropped_lr_model = mult_lin_reg(df_dropped_cols, lr3_input_features, lr_label)
    model_stats(df_dropped_cols, dropped_lr_model, "R2", "CSE", "DoF", "MSE", "RSS","t-Values", "p-Values")

    # calculate vif on original dataframe
    vif_orig_df = vif(orig_df, True)
