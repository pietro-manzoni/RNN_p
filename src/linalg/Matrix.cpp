/*
Author: P.Manzoni & R.Baviera
Last Modified: P.M., 22-05-2025
Copyright (c) 2025, Politecnico di Milano, QFinLab
*/
#include "Matrix.h"
#include <iostream>
#include <iomanip>
#include <random>


/********************************************** CONSTRUCTORS ************************************************/

// Initialization with a single given value, equal for all entries
Matrix::Matrix(unsigned rows_, unsigned cols_, double value_) :
        elements( std::vector<double>(rows_ * cols_, value_) ),
        ROWS(rows_), COLS(cols_), SIZE(rows_ * cols_) {}


/********************************************** OPERATORS ***************************************************/

// Assignment operator
Matrix& Matrix::operator=(const Matrix& rhs){

    if (this != &rhs) { // if we are trying to assign a Matrix to itself, we just return the original matrix

        // check if the two matrices have compatible sizes
        if ((ROWS!=rhs.ROWS) || (COLS!=rhs.COLS)) {
            std::cerr << "Error: Trying to assign matrices with incompatible sizes!" << std::endl;
            exit(EXIT_FAILURE);
        }

        // if compatible, copy elements
        elements = rhs.elements;

    }
    return *this;
}


// Elementwise access operator (Return Copy)
double Matrix::operator()(unsigned i, unsigned j) const{
    if ((i >= ROWS) || (j >= COLS)){
        std::cerr << "Error in matrix operator () : idx out of bounds" << std::endl;
        exit(EXIT_FAILURE);
    }
    return elements[i * COLS + j];
}


// Elementwise access operator (Return Reference)
double& Matrix::operator()(unsigned i, unsigned j){
    if ((i >= ROWS) || (j >= COLS)){
        std::cerr << "Error in matrix operator () : idx out of bounds" << std::endl;
        exit(EXIT_FAILURE);
    }
    return elements[i * COLS + j];
}


// Elementwise matrix sum
Matrix& Matrix::operator+=(const Matrix& rhs){
    if ((ROWS != rhs.ROWS) || (COLS != rhs.COLS)){
        std::cerr << "Error in performing matrix += operation: incompatible sizes" << std::endl;
        exit(EXIT_FAILURE);
    }
    for (unsigned i = 0; i < SIZE; ++i)
        elements[i] += rhs.elements[i];
    return (*this);
}


// Elementwise matrix difference
Matrix& Matrix::operator-=(const Matrix& rhs){
    if ((ROWS != rhs.ROWS) || (COLS != rhs.COLS)){
        std::cerr << "Error in performing matrix -= operation: incompatible sizes" << std::endl;
        exit(EXIT_FAILURE);
    }
    for (unsigned i = 0; i < SIZE; ++i)
        elements[i] -= rhs.elements[i];
    return (*this);
}


/********************************************** METHODS *****************************************************/

double Matrix::loc(unsigned n) const{
    if ((n >= SIZE)){
        std::cerr << "Error in matrix method 'flat': idx out of bounds" << std::endl;
        exit(EXIT_FAILURE);
    }
    return elements[n];
}

double& Matrix::loc(unsigned n){
    if ((n >= SIZE)){
        std::cerr << "Error in matrix method 'flat': idx out of bounds" << std::endl;
        exit(EXIT_FAILURE);
    }
    return elements[n];
}

// Print Matrix
void Matrix::print() const{
    for (unsigned i = 0; i < ROWS; ++i) {
        for (unsigned j = 0; j < COLS; ++j)
            std::cout << std::right << std::setw(12) << elements[i*COLS + j] ;
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Set all Matrix entries to zero
void Matrix::to_zero(){
    for (unsigned i = 0; i < SIZE; ++i)
        elements[i] = 0;
}

// Generate transpose matrix
Matrix Matrix::transpose() const{

    // we create a matrix of the opposite shape
    Matrix new_matrix(COLS, ROWS, 0.);

    // we assign all elements of the original matrix to the new one
    for (unsigned i = 0; i < ROWS; ++i)
        for (unsigned j = 0; j < COLS; ++j)
            new_matrix(j, i) = (*this)(i, j);

    return new_matrix;
}
