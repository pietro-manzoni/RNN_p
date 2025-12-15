/*
Author: P.Manzoni & R.Baviera
Last Modified: P.M., 22-05-2025
Copyright (c) 2025, Politecnico di Milano, QFinLab
*/

#ifndef RNN_p_ALGEBRAICOPERATIONS_H
#define RNN_p_ALGEBRAICOPERATIONS_H


#include "Matrix.h"
#include "Vector.h"

/************************ ELEMENTWISE OPERATIONS BETWEEN TWO VECTORS *************************************/

/// Addition of two vectors
Vector operator+(const Vector& vec1, const Vector& vec2);

/// Subtraction of two vectors
Vector operator-(const Vector& vec1, const Vector& vec2);

/// Division of two vectors
Vector operator/(const Vector& vec1, const Vector& vec2);


/**************************** OPERATIONS BETWEEN A VECTOR and A SCALAR **********************************/

/// Addition of vector and scalar
Vector operator+(const Vector& vec, double scalar);

/// Subtraction between vector and scalar
Vector operator-(const Vector& vec, double scalar);

/// Multiplication between vector and scalar
Vector operator*(const Vector& vec, double scalar);

/// Division between vector and scalar
Vector operator/(const Vector& vec, double scalar);


/********************************** NONLINEAR TRANSFORMATIONS of a VECTOR ***************************************/

/// Square power of vector
Vector pow2(const Vector& vec);

/// Square root of vector
Vector sqrt(const Vector& vec);

/// Exponential of vector
Vector exp(const Vector& vec);


/**************************** MULTIPLICATION BETWEEN TWO MATRICES  **************************************/

/// Matrix-Matrix multiplication
Matrix operator*(const Matrix& mat1, const Matrix& mat2);

/// Matrix-Matrix multiplication with no-aliasing
/**
 * Multiplication operator that reveals to be faster for medium-large matrices.
 * Usually the result of the multiplication of two matrices is stored in a
 * temporary matrix and then assigned to the first member through the assignment
 * operator, which involves a further copy of the matrix itself. In this case the results
 *  are directly stored in the #where matrix.
 * @param mat1:     LHS of the multiplication
 * @param mat2:     RHS of the multiplication
 * @param where:    Matrix where the result has to be stored
 */
void mult_in_place(const Matrix& mat1, const Matrix& mat2, Matrix& where);


/*********************** MULTIPLICATION BETWEEN a MATRIX and a COLUMN VECTOR  ***********************************/

/// Matrix-Vector multiplication
Vector operator*(const Matrix& mat, const Vector& vec);

/// Matrix-Vector multiplication with no-aliasing
/**
 * Multiplication operator that reveals to be faster for medium-large matrices.
 * Usually the result of the multiplication of matrix-vector is stored in a
 * temporary vector and then assigned to the first member through the assignment
 * operator, which involves a further copy of the vector itself. In this case the results
 * are directly stored in the #where matrix.
 * @param mat:      LHS of the multiplication
 * @param vec:      RHS of the multiplication
 * @param where:    vector where the result has to be stored
 */
void mult_in_place(const Matrix& mat, const Vector& vec, Vector& where);


/*********************** MULTIPLICATION BETWEEN a ROW VECTOR and a MATRIX  ***********************************/

/// Vector-Matrix multiplication with no-aliasing
/**
 * \note The returned vector should be interpreted as \a row \a vector
 */
Vector operator*(const Vector& vec, const Matrix& mat);

/// Vector-Matrix multiplication with no-aliasing
/**
 * Multiplication operator that reveals to be faster for medium-large matrices.
 * Usually the result of the multiplication of matrix-vector is stored in a
 * temporary vector and then assigned to the first member through the assignment
 * operator, which involves a further copy of the vector itself. In this case the results
 * are directly stored in the #where matrix.
 * \note The returned vector should be interpreted as \a row \a vector
 * @param vec:      LHS of the multiplication
 * @param mat:      RHS of the multiplication
 * @param where:    vector where the result has to be stored
 */
void mult_in_place(const Vector& vec, const Matrix& mat, Vector& where);

/******************* MULTIPLICATION BETWEEN a FULL MATRIX and a DIAGONAL MATRIX  ********************************/

/// Optimized product A * D, with A full matrix and D diagonal matrix (stored as a vector)
Matrix mult_diagonal(const Matrix& mat, const Vector& diag);

/// Optimized product A * D, with A full matrix and D diagonal matrix (stored as a vector)
void mult_diagonal(const Matrix& mat, const Vector& diag, Matrix& where);


#endif //RNN_p_ALGEBRAICOPERATIONS_H
