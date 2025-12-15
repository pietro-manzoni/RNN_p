/*
Author: P.Manzoni & R.Baviera
Last Modified: P.M., 22-05-2025
Copyright (c) 2025, Politecnico di Milano, QFinLab
*/

#include "Vector.h"
#include <iostream>
#include <iomanip>
#include <random>

/********************************************** CONSTRUCTORS ************************************************/

// Initialization with a single given value, equal for all entries
Vector::Vector(unsigned size_, double value_) :
        elements( std::vector<double>(size_, value_) ), SIZE(size_) {}


/********************************************** OPERATORS ***************************************************/

// Assignment operator
Vector& Vector::operator=(const Vector& rhs){

    if (this != &rhs) { //if we are trying to assign a Vector to itself, we just return the original Vector

        // check if the two vectors have compatible sizes
        if (SIZE!=rhs.SIZE) {
            std::cerr << "Error: Trying to assign Vectors with incompatible sizes!" << std::endl;
            exit(EXIT_FAILURE);
        }

        // if compatible, copy elements
        elements = rhs.elements;

    }
    return *this;
}


// Elementwise access operator (Return Copy)
double Vector::operator()(unsigned i) const{
    if (i >= SIZE){
        std::cerr << "Error in Vector operator () : idx out of bounds" << std::endl;
        exit(EXIT_FAILURE);
    }
    return elements[i];
}


// Elementwise access operator (Return Reference)
double& Vector::operator()(unsigned i){
    if (i >= SIZE){
        std::cerr << "Error in Vector operator () : idx out of bounds" << std::endl;
        exit(EXIT_FAILURE);
    }
    return elements[i];
}


// Elementwise Vector sum
Vector& Vector::operator+=(const Vector& rhs){
    if (SIZE != rhs.SIZE){
        std::cerr << "Error in performing Vector += operation: incompatible sizes" << std::endl;
        exit(EXIT_FAILURE);
    }
    for (unsigned i = 0; i < SIZE; ++i)
        elements[i] += rhs.elements[i];
    return (*this);
}


// Elementwise Vector difference
Vector& Vector::operator-=(const Vector& rhs){
    if (SIZE != rhs.SIZE){
        std::cerr << "Error in performing Vector -= operation: incompatible sizes" << std::endl;
        exit(EXIT_FAILURE);
    }
    for (unsigned i = 0; i < SIZE; ++i)
        elements[i] -= rhs.elements[i];
    return (*this);
}


// Elementwise Vector multiplication by scalar
Vector& Vector::operator*=(double scalar){
    for (unsigned i = 0; i < SIZE; ++i)
        elements[i] *= scalar;
    return (*this);
}


// Elementwise Vector division by scalar
Vector& Vector::operator/=(double scalar){
    for (unsigned i = 0; i < SIZE; ++i)
        elements[i] /= scalar;
    return (*this);
}


/********************************************** METHODS *****************************************************/

// Print Vector
void Vector::print(bool as_row) const{
    // print as ROW vector
    if ( as_row ) {
        for (unsigned i = 0; i < SIZE; ++i)
            std::cout << std::right << std::setw(12) << elements[i];
        std::cout << "\n" << std::endl;
    }
        // print as COLUMN vector
    else {
        for (unsigned i = 0; i < SIZE; ++i)
            std::cout << std::right << std::setw(12) << elements[i] << "\n";
        std::cout << std::endl;
    }
}

// Set all Vector entries to zero
void Vector::to_zero(){
    for (unsigned i = 0; i < SIZE; ++i)
        elements[i] = 0;
}
