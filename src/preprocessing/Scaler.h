/*
Author: R.Baviera & P.Manzoni
Last Modified: P.M., 22-05-2025
Copyright (c) 2025, Politecnico di Milano, QFinLab
*/


#ifndef RNN_p_SCALER_H
#define RNN_p_SCALER_H

#include <vector>
#include <algorithm>
#include <iostream>

#include "dataframe/DataFrame.h"
#include "linalg/Vector.h"


class Scaler {

// ===================================
//   Attributes
// ===================================

private:
    std::vector<std::string> columns;       // Name of each processed feature
    std::vector<double> min_vals;           // Minimum values for each feature (column)
    std::vector<double> max_vals;           // Maximum values for each feature (column)


// ===================================
//   Fit and transform
// ===================================

public:
    // Method to fit the scaler (find min and max for each feature)
    void fit(const DataFrame& df);

    // Method to transform the data (normalize each feature independently if required)
    void transform(DataFrame& df) const;

    // Method to get the minimum and the maximum given a column
    std::pair<double, double> get_minmax(const std::string& col) const;
};


#endif //RNN_p_SCALER_H