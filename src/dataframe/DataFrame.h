/*
Author: P.Manzoni & R.Baviera
Last Modified: P.M., 22-05-2025
Copyright (c) 2025, Politecnico di Milano, QFinLab
*/

#ifndef RNN_p_DATAFRAME_H
#define RNN_p_DATAFRAME_H

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>


class DataFrame {

public:

    // Internal state (kept private)
    std::vector<std::string> index;
    std::vector<std::string> columns;
    std::vector<std::vector<double>> data;
    std::string index_name;

    // constructors
    explicit DataFrame(const std::string& csv_path);
    DataFrame(const std::vector<std::string>& index, const std::vector<std::string>& columns);

    // copy constructor
    DataFrame(const DataFrame&) = default;

    // access elements
    double& at(const std::string& row_label, const std::string& col_label);
    [[nodiscard]] const double& at(const std::string& row_label, const std::string& col_label) const;

    // filter rows by index range
    void filter_by_time_range(const std::string& start, const std::string& end);

    // filter rows by hour, i.e. keep only the specified hour
    void filter_by_hour(unsigned target_hour);

    // keep only selected columns
    void filter_by_columns(const std::vector<std::string>& selected_columns);

    // apply logarithm to the selected columns
    void apply_log(const std::vector<std::string>& selected_columns);

    // print to csv
    void to_csv(const std::string& filename) const;


protected:

    unsigned find_column(const std::string& name) const;

};


#endif //RNN_p_DATAFRAME_H
