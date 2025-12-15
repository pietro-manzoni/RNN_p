/*
Author: P.Manzoni & R.Baviera
Last Modified: P.M., 22-05-2025
Copyright (c) 2025, Politecnico di Milano, QFinLab

The important feature to notice is that the Matrix class (similarly to the Vector class) has immutable size.
This makes them very robust for the purpose, but not very versatile as general objects (e.g. for other projects).
*/

#ifndef RNN_p_MATRIX_H
#define RNN_p_MATRIX_H

#include <vector>

class Matrix {

    /********************************************** ATTRIBUTES **************************************************/

protected:

    /// Elements of the Matrix
    std::vector<double> elements;


public:

    /// Number of rows of the Matrix
    const unsigned ROWS;

    /// Number of columns of the Matrix
    const unsigned COLS;

    /// Number of elements in the Matrix
    /**
     * It coincides with the length of the vector data and is equal to #n_rows * #n_cols.
     */
    const unsigned SIZE;

    /****************************************** CONSTRUCTORS ********************************************/

public:

    /// Constructor: initialization with a single given value
    /**
     * Allocate an array of given size in the heap and initialize all the elements
     * with a given value (default is 0).
     * @param rows_:        required row size
     * @param cols_:        required column size
     * @param value_:       initializer value
     */
    Matrix(unsigned rows_, unsigned cols_, double value_ = 0);


    /******************************************* DESTRUCTORS **********************************************/

public:

    /// Destructor of the object (default, as no dynamic objects are involved)
    ~Matrix() = default;


    /****************************************** OPERATORS ************************************************/

public:

    /// Assignment operator
    /**
     * Assign the RHS to the LHS. The array #components is
     * copied and not shared, so the two objects are independent.
     * @param rhs: matrix to be copied
     * @return reference to the copied matrix
     */
    Matrix& operator=(const Matrix& rhs);

    /// Access operator (Return Copy)
    /**
     * Access the (i,j) element of the matrix
     * @param i: row-index in the array
     * @param j: column-index in the array
     * @return copy of the (i,j) element of the matrix
     */
     double operator()(unsigned i, unsigned j) const;

    /// Access operator (Return Reference)
    /**
     * Access the (i,j) element of the matrix
     * @param i: row-index in the array
     * @param j: column-index in the array
     * @return reference to the (i,j) element of the matrix
     */
    double& operator()(unsigned i, unsigned j);

    /// In-place elementwise sum operator
    /**
     * In-place elementwise sum
     * @param rhs: matrix to be added
     * @return reference to the obtained matrix
     */
    Matrix& operator+=(const Matrix& rhs);

    /// In-place elementwise difference operator
    /**
     * In-place elementwise subtraction
     * @param rhs: matrix to be subtracted
     * @return reference to the obtained matrix
     */
    Matrix& operator-=(const Matrix& rhs);


    /********************************************** METHODS *****************************************************/

public:

    /// Get access to the n-th element, ordered by rows (Return Copy)
    /**
     * Access the n-th element
     * @param n: index in the std::vector %elements
     * @return copy of the n-th entry of the std::vector %elements
     */
    double loc(unsigned n) const;

    /// Get access to the i-th element, ordered by rows (Return Reference)
    /**
     * Access the (i,j) element of the matrix
     * @param n: index in the array
     * @return reference to the n-th entry of the std::vector %elements
     */
    double& loc(unsigned n);

    /// Print
    void print() const;

    /// Set all entries to zero, while preserving the size of the Matrix
    void to_zero();

    /// Transpose
    Matrix transpose() const;

};


#endif //RNN_p_MATRIX_H
