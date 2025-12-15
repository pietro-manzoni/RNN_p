/*
Author: P.Manzoni & R.Baviera
Last Modified: P.M., 22-05-2025
Copyright (c) 2025, Politecnico di Milano, QFinLab
*/

#ifndef RNN_p_LINEARMODEL_H
#define RNN_p_LINEARMODEL_H


#include <vector>
#include <iostream>

#include "linalg/Matrix.h"
#include "linalg/Vector.h"



class LinearModel {

    /********************************************** ATTRIBUTES **************************************************/

private:

    Vector betas;


    /****************************************** CONSTRUCTORS ********************************************/

public:

    /// Default constructor
    LinearModel(const std::vector<std::vector<double>>& exog, const std::vector<std::vector<double>>& endog);


    /********************************************** METHODS *****************************************************/

private:

  /// Solve a linear system using LUP factorization
  /**
   * Solve a linear system employing the LU (Lower-Upper triangular) factorization
   * of the matrix with pivotal Permutation.
   * @param A:      design matrix of linear model
   * @param b:      regressand variable
   */
    static Vector LUP_solve(Matrix& A, Vector& b);

    /// Adapt matrix A so that outliers are not considered in linear regression (OLS)
    /**
     * First, vector b is analysed. The ouliers (identified according to IQR criterion)
     * are spotted and their indexes stored. Then each row of the design matrix
     * that corresponds to an outlier is set to 0. This can be proved to be
     * mathematically equivalent (in terms of OLS projection) to the removal of these rows.
     *
     * @param A:      design matrix of linear model
     * @param b:      regressand variable
     */
    static void outliers_removal(Matrix& A, Vector& b);


public:

    /// Print betas of the regression
    void params(void) const;

    /// Predict with linear model
    std::vector<double> predict(const std::vector<std::vector<double>>& exog);

};


#endif //RNN_p_LINEARMODEL_H
