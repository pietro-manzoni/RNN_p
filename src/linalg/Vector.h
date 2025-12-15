/*
Author: P.Manzoni & R.Baviera
Last Modified: P.M., 22-05-2025
Copyright (c) 2025, Politecnico di Milano, QFinLab

The important feature to notice is that the Vector class (similarly to the Matrix class) has immutable size.
This makes them very robust for the purpose, but not very versatile as general objects (e.g. for other projects).
*/


#ifndef RNN_p_VECTOR_H
#define RNN_p_VECTOR_H

#include <vector>

class Vector {

    /********************************************** ATTRIBUTES **************************************************/

protected:

    /// Elements of the Vector
    std::vector<double> elements;

public:

    /// Number of elements in the Vector
    const unsigned SIZE;

    /****************************************** CONSTRUCTORS ********************************************/

public:

    /// Constructor: initialization with a single given value
    /**
     * Allocate an array of given size and initialize all the elements
     * with a given value (default is 0).
     * @param size_:   size of the array
     * @param value_:   initializer value
     */
    explicit Vector(unsigned size_, double value_ = 0);


    /******************************************* DESTRUCTORS **********************************************/

public:

    /// Destructor of the object (default, as no dynamic objects are involved)
    ~Vector() = default;


    /****************************************** OPERATORS ************************************************/

public:

    /// Assignment operator
    /**
     * Assign the RHS to the LHS.
     * @param rhs: vector to be copied
     * @return reference to the copied vector
     */
    Vector& operator=(const Vector& rhs);

    /// Access operator (Return Copy)
    /**
     * Access the i-th element of the #components Vector.
     * \note No check on the correctness of the index is performed
     * @param idx: index in the array
     * @return copy of the i-th element of the Vector
     */
    double operator()(unsigned idx) const;

    /// Access operator (Return Reference)
    /**
     * Access the i-th element of the #components Vector.
     * @param idx: index in the array
     * @return reference to the i-th element of the Vector
     */
    double& operator()(unsigned idx);


    /// In-place elementwise sum operator
    /**
     * In-place elementwise sum
     * @param rhs: Vector to be added
     * @return reference to the obtained Vector
     */
    Vector& operator+=(const Vector& rhs);

    /// In-place elementwise difference operator
    /**
     * In-place elementwise subtraction
     * @param rhs: Vector to be subtracted
     * @return reference to the obtained Vector
     */
    Vector& operator-=(const Vector& rhs);

    /// In-place product operator
    /**
     * In-place elementwise multiplication
     * @param scalar: multiplication coefficient
     * @return reference to the obtained vector
     */
     Vector& operator*=(double scalar);

    /// In-place division operator
    /**
     * In-place elementwise division
     * @param scalar: division coefficient
     * @return reference to the obtained vector
     */
     Vector& operator/=(double scalar);

    /********************************************** METHODS *****************************************************/

public:

    /// Print
    void print(bool as_row = true) const;

    /// Set all entries to zero, while preserving the size of the Matrix
    void to_zero();

};


#endif //RNN_p_VECTOR_H
