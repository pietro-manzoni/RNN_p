/*
Author: R.Baviera & P.Manzoni
Last Modified: P.M., 22-05-2025
Copyright (c) 2025, Politecnico di Milano, QFinLab
*/

#include "Scaler.h"

// Method to fit the scaler (find min and max for each feature)
void Scaler::fit(const DataFrame& df) {

    // we alias the content of the dataframe for brevity
    const auto& data = df.data;

    // initial check
    if (data.empty() || data[0].empty()) {
        std::cerr << "Error: Input DataFrame is empty.\n";
        exit(EXIT_FAILURE);
    }


    // extract the number of features
    unsigned num_features = data[0].size();

    // resize accordingly the members min_vals and max_vals
    min_vals.resize(num_features);
    max_vals.resize(num_features);

    // copy the name of the columns
    columns = df.columns;

    // find min and max for each feature (column) to normalize
    for (unsigned i = 0; i < num_features; ++i) {

        // initialize the value with the first available one
        min_vals[i] = data[0][i];
        max_vals[i] = data[0][i];

        // if a new max/min is found, overwrite the existing one
        for (const auto& row : data) {
            min_vals[i] = std::min(min_vals[i], row[i]);
            max_vals[i] = std::max(max_vals[i], row[i]);
        }

        // display warning if maximum and minimum coincide
        if (max_vals[i] == min_vals[i])
            std::cerr << "Warning: Skipping normalization for feature " << i
                      << " because max == min (" << max_vals[i] << ")\n";

    }

}


// Method to transform the data (normalize each feature independently if required)
void Scaler::transform(DataFrame& df) const{
    for (auto& row : df.data)
        for (unsigned i = 0; i < row.size(); ++i)
            if (max_vals[i] != min_vals[i])
                row[i] = (row[i] - min_vals[i]) / (max_vals[i] - min_vals[i]);
}


std::pair<double, double> Scaler::get_minmax(const std::string& col) const{

    // Search for the column in the stored list
    for (size_t i = 0; i < columns.size(); ++i) {
        if (columns[i] == col) {
            return {min_vals[i], max_vals[i]};
        }
    }

    // if the column is not found, throw an error
    throw std::invalid_argument("Scaler::get_minmax: column \"" + col + "\" not found.");
}