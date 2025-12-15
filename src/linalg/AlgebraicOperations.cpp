/*
Author: P.Manzoni & R.Baviera
Last Modified: P.M., 22-05-2025
Copyright (c) 2025, Politecnico di Milano, QFinLab
*/

#include <cmath>
#include <iostream>

#include "AlgebraicOperations.h"

/******************************* SUM and DIFFERENCE BETWEEN TWO VECTORS ******************************************/

Vector operator+(const Vector& vec1, const Vector& vec2){
    if (vec1.SIZE != vec2.SIZE){
        std::cerr << "Wrong size for Vector summation" << std::endl;
        exit(EXIT_FAILURE);
    }

    Vector out_vec(vec1.SIZE);
    for (unsigned i = 0; i < vec1.SIZE; ++i)
        out_vec(i) = vec1(i) + vec2(i);
    return out_vec;
}

Vector operator-(const Vector& vec1, const Vector& vec2){
    if (vec1.SIZE != vec2.SIZE){
        std::cerr << "Wrong size for vector difference" << std::endl;
        exit(EXIT_FAILURE);
    }

    Vector out_vec(vec1.SIZE);
    for (unsigned i = 0; i < vec1.SIZE; ++i)
        out_vec(i) = vec1(i) - vec2(i);
    return out_vec;
}

Vector operator/(const Vector& vec1, const Vector& vec2){
    if (vec1.SIZE != vec2.SIZE){
        std::cerr << "Wrong size for vector difference" << std::endl;
        exit(EXIT_FAILURE);
    }

    Vector out_vec(vec1.SIZE);
    for (unsigned i = 0; i < vec1.SIZE; ++i)
        out_vec(i) = vec1(i) / vec2(i);
    return out_vec;
}


/**************************** OPERATIONS BETWEEN a VECTOR and a SCALAR **********************************/

Vector operator+(const Vector& vec, double scalar){
    Vector out_vec(vec.SIZE);
    for (unsigned i = 0; i < vec.SIZE; ++i)
        out_vec(i) = vec(i) + scalar;
    return  out_vec;
}

Vector operator-(const Vector& vec, double scalar){
    Vector out_vec(vec.SIZE);
    for (unsigned i = 0; i < vec.SIZE; ++i)
        out_vec(i) = vec(i) - scalar;
    return  out_vec;
}

Vector operator*(const Vector& vec, double scalar){
    Vector out_vec(vec.SIZE);
    for (unsigned i = 0; i < vec.SIZE; ++i)
        out_vec(i) =  vec(i) * scalar;
    return  out_vec;
}

Vector operator/(const Vector& vec, double scalar){
    Vector out_vec(vec.SIZE);
    for (unsigned i = 0; i < vec.SIZE; ++i)
        out_vec(i) = vec(i) / scalar;
    return  out_vec;
}

/********************************** NONLINEAR TRANSFORMATIONS of a VECTOR ***************************************/

Vector pow2(const Vector& vec){
    Vector out_vec(vec.SIZE);
    for (unsigned i = 0; i < vec.SIZE; ++i)
        out_vec(i) = vec(i) * vec(i);
    return  out_vec;
}

Vector sqrt(const Vector& vec){
    Vector out_vec(vec.SIZE);
    for (unsigned i = 0; i < vec.SIZE; ++i)
        out_vec(i) = std::sqrt( vec(i) );
    return  out_vec;
}

Vector exp(const Vector& vec) {
    Vector out_vec(vec.SIZE);
    for (unsigned i = 0; i < vec.SIZE; ++i)
        out_vec(i) = std::exp(vec(i));
    return out_vec;
}


/**************************** MULTIPLICATION BETWEEN TWO MATRICES  **************************************/

Matrix operator*(const Matrix& mat1, const Matrix& mat2){

    if (mat1.COLS != mat2.ROWS){
        std::cerr << "Wrong size for Matrix-Matrix multiplication" << std::endl;
        exit(EXIT_FAILURE);
    }

    Matrix out(mat1.ROWS, mat2.COLS, 0.);
    for (unsigned i = 0; i < out.ROWS; ++i)
        for (unsigned k = 0; k < mat1.COLS; ++k)
            for (unsigned j = 0; j < out.COLS; ++j) // swapped order of j-loop and k-loop to increase cache-ability //TODO: check se è vero
                out(i, j) += mat1(i, k) * mat2(k, j);

    return out;
}


void mult_in_place(const Matrix& mat1, const Matrix& mat2, Matrix& where){

    if (mat1.COLS != mat2.ROWS){
        std::cerr << "Wrong size for Matrix-Matrix multiplication" << std::endl;
        exit(EXIT_FAILURE);
    }
    if ((mat1.ROWS != where.ROWS) || (mat2.COLS!= where.COLS)){
        std::cerr << "Wrong size for WHERE matrix" << std::endl;
        exit(EXIT_FAILURE);
    }

    where.to_zero();

    for (unsigned i = 0; i < where.ROWS; ++i)
        for (unsigned k = 0; k < mat1.COLS; ++k)
            for (unsigned j = 0; j < where.COLS; ++j) // swapped order of j-loop and k-loop to increase cache-ability
                where(i,j) += mat1(i,k) * mat2(k,j);

}


/*********************** MULTIPLICATION BETWEEN a MATRIX and a COLUMN VECTOR  ***********************************/

Vector operator*(const Matrix& mat, const Vector& vec){

    if (mat.COLS != vec.SIZE){
        std::cerr << "Wrong size for Matrix-Vector multiplication" << std::endl;
        exit(EXIT_FAILURE);
    }

    Vector out(mat.ROWS, 0.);
    for (unsigned i = 0; i < out.SIZE; ++i)
        for (unsigned j = 0; j < mat.COLS; ++j)
            out(i) += mat(i,j) * vec(j);

    return  out;
}


void mult_in_place(const Matrix& mat, const Vector& vec, Vector& where){

    if (mat.COLS != vec.SIZE){
        std::cerr << "Wrong size for Matrix-Vector multiplication" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (mat.ROWS != where.SIZE){
        std::cerr << "Wrong size for WHERE Vector" << std::endl;
        exit(EXIT_FAILURE);
    }

    where.to_zero();

    for (unsigned i = 0; i < where.SIZE; ++i)
        for (unsigned j = 0; j < mat.COLS; ++j)
            where(i) += mat(i,j) * vec(j);

}


/*********************** MULTIPLICATION BETWEEN a ROW VECTOR and a MATRIX  ***********************************/

Vector operator*(const Vector& vec, const Matrix& mat){

    if (vec.SIZE != mat.ROWS){
        std::cerr << "Wrong size for Vector-Matrix multiplication" << std::endl;
        exit(EXIT_FAILURE);
    }

    Vector out(mat.COLS);
    for (unsigned j = 0; j < out.SIZE; ++j){
        for (unsigned i = 0; i < vec.SIZE; ++i)
            out(j) += vec(i) * mat(i,j);
    }

    return  out;
}


void mult_in_place(const Vector& vec, const Matrix& mat, Vector& where) {

    if (vec.SIZE != mat.ROWS) {
        std::cerr << "Wrong size for Matrix-Vector multiplication" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (mat.COLS != where.SIZE) {
        std::cerr << "Wrong size for WHERE Vector" << std::endl;
        exit(EXIT_FAILURE);
    }

    where.to_zero();

    for (unsigned j = 0; j < where.SIZE; ++j)
        for (unsigned i = 0; i < vec.SIZE; ++i)
            where(j) += vec(i) * mat(i, j);

}


/******************* MULTIPLICATION BETWEEN a FULL MATRIX and a DIAGONAL MATRIX  ********************************/

Matrix mult_diagonal(const Matrix& mat, const Vector& diag){

    if (mat.COLS != diag.SIZE){
        std::cerr << "Wrong size for Matrix-Matrix multiplication" << std::endl;
        exit(EXIT_FAILURE);
    }

    Matrix out(mat.ROWS, mat.COLS, 0.);

    for (unsigned i = 0; i < out.ROWS; ++i)
        for (unsigned j = 0; j < out.COLS; ++j)
            out(i,j) = mat(i,j) * diag(j);

    return out;

}

void mult_diagonal(const Matrix& mat, const Vector& diag, Matrix& where){

    if (mat.COLS != diag.SIZE){
        std::cerr << "Wrong size for Matrix-Matrix multiplication" << std::endl;
        exit(EXIT_FAILURE);
    }
    if ((mat.ROWS != where.ROWS) || diag.SIZE != where.COLS){
        std::cerr << "Wrong size for WHERE matrix" << std::endl;
        exit(EXIT_FAILURE);
    }

    where.to_zero();

    for (unsigned i = 0; i < where.ROWS; ++i)
        for (unsigned j = 0; j < diag.SIZE; ++j)
            where(i,j) = mat(i,j) * diag(j);
}
