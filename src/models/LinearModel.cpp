/*
Author: P.Manzoni & R.Baviera
Last Modified: P.M., 22-05-2025
Copyright (c) 2025, Politecnico di Milano, QFinLab
*/

#include <cmath>
#include <algorithm>

#include "linalg/AlgebraicOperations.h"
#include "models/LinearModel.h"

/********************************************** METHODS *****************************************************/

void LinearModel::params(void) const{

    std::cout << "Betas for Linear Model:" << std::endl;
    betas.print();
    std::cout << std::endl;

}

/// Default constructor
LinearModel::LinearModel(const std::vector<std::vector<double>>& exog, const std::vector<std::vector<double>>& endog) :
    betas( Vector(exog[0].size()+1) )
{
    // creating design matrix X of regressors (by copying the std::vector<std::vector<double>>)
    Matrix X(exog.size(), exog[0].size()+1);
        for (unsigned i = 0; i < X.ROWS; ++i) {
            X(i, 0) = 1.; // intercept
            for (unsigned j = 1; j < X.COLS; ++j)
                X(i, j) = exog[i][j-1];
        }

    // creating vector y of target variable
    Vector y(endog.size());
    for (unsigned i = 0; i < y.SIZE; ++i)
        y(i) = endog[i][0];

    // if requested, we remove the outliers
    outliers_removal(X, y);

    // creating normal equations (X^T X beta = Z^T y)
    Matrix X_t = X.transpose();
    Matrix A = X_t * X;
    Vector b = X_t * y;

    // finding betas
    betas = LUP_solve(A, b);

}


std::vector<double> LinearModel::predict(const std::vector<std::vector<double>>& exog){

    // creating design matrix X of regressors (from std::vector<std::vector<double>>)
    Matrix X(exog.size(), exog[0].size()+1);
    for (unsigned i = 0; i < X.ROWS; ++i) {
        X(i, 0) = 1.; // intercept
        for (unsigned j = 1; j < X.COLS; ++j)
            X(i, j) = exog[i][j-1];
    }

    // predicting
    Vector prediction_vec = X * betas;

    // converting output value
    std::vector<double> prediction(prediction_vec.SIZE);
    for (unsigned i = 0; i < prediction_vec.SIZE; ++i)
        prediction[i] = prediction_vec(i);

    return prediction;

}


Vector LinearModel::LUP_solve(Matrix& A, Vector& b){

    // sanity check
    if (A.ROWS != b.SIZE){
        std::cerr << "Matrix and vector size are not compatible" << std::endl;
        exit(3);
    }

    const double TOL = 1e-10;
    /* Tolerance for considering a matrix as SINGULAR: if the greatest available pivot (that can be chosen with a
     * row permutation) is less than TOL in absolute value, matrix is declared singular and it is not decomposed */

    const unsigned N = A.ROWS; //cache value
    Vector P(N,0);  //initialize permutation vector
    Vector x(N,0);  //initialize solution vector


    // LUP decomposition

    for (unsigned i = 0; i < N; i++)
        P(i) = i; //initialize P sequentially, it represents the actual order in which rows are processed

    for (unsigned i = 0; i < N; i++) {
        double maxA = 0, absA = 0;
        unsigned imax = i; // ideally, I would work on the (i,i) pivot (but maybe a permutation will take place)

        // having fixed the i-th column, I choose which is the k-th row (k>=i) that should be processed in order
        // to have the greatest possible pivot (in absolute value)
        for (unsigned k = i; k < N; k++)
            if ( (absA = std::fabs(A(k,i)) ) > maxA) {
                maxA = absA;    // maximum value of A(k,i) for k varying in [i,N)
                imax = k;       // row for which the maximum is attained
            }

        if (maxA < TOL) {
            std::cerr << "Matrix is too close to singular" << std::endl;
            exit(3); // The maximum available pivot is too small. Exit.
        }

        // if the choice of the pivot requires a permutation, I keep track of this
        if (imax != i)
            std::swap( P(imax), P(i) );

        // use Gauss elimination method
        for (unsigned j = i + 1; j < N; j++) {
            A( P(j), i) /= A( P(i), i);
            for (unsigned k = i + 1; k < N; k++)
                A( P(j), k) -= A( P(j), i) * A( P(i), k);
        }

    }

    // At this point the matrix A is dense and contains:
    // - the U matrix in its upper triangular part (including the diagonal)
    // - the L matrix in its lower triangular part (excluding the diagonal, which is unary by definition)
    // P contains the permutation order of the rows

    // FORWARD substitution
    for (unsigned i = 0; i < N; ++i) {
        x(i) = b(P(i));
        for (unsigned k = 0; k < i; ++k)
            x(i) -= A(P(i),k) * x(k);
    }

    // BACKWARD substitution
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) { //because i>=0, 'unsigned' type cannot be used
        for (unsigned k = i + 1; k < N; ++k)
            x(i) -= A(P(i),k) * x(k);
        x(i) = x(i) / A(P(i),i);
    }

    return x;

}


void LinearModel::outliers_removal(Matrix& A, Vector& b){
/* to avoid leverage effect in linear model, we want to remove the rows corresponding to outliers in the training
 * regressand. It is easy to show that, using OLS, this is equivalent to setting to zero the selected rows in the
 * design matrix. Outliers are selected by means of InterQuantile Range (IQR) */

    // create a copy of b (a std::vector)
    std::vector<double> b_vec(b.SIZE);
    for (unsigned i = 0; i < b.SIZE; ++i)
        b_vec[i] = b(i);

    //sort the copy
    std::sort(b_vec.begin(), b_vec.end());

    // find 1st and 3rd quartiles
    const double Q1 = b_vec[  b_vec.size()/4];
    const double Q3 = b_vec[3*b_vec.size()/4];
    const double IQR = Q3 - Q1;

    // find outliers and set corresponding row of design matrix to 0
    for (unsigned i = 0; i < b.SIZE; ++i)
        if ((b(i) > Q3 + 3 * IQR) || (b(i) < Q1 - 3 * IQR)){
            for (unsigned j = 0; j < A.COLS; ++j)
                A(i, j) = 0;
        //std::cout << i << std::endl;
        }
}
