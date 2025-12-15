/*
Author: P.Manzoni & R.Baviera
Last Modified: P.M., 22-05-2025
Copyright (c) 2025, Politecnico di Milano, QFinLab
*/

#include <fstream>
#include <sstream>
#include <functional>
#include <limits>
#include <cmath>
#include <algorithm>

#include "DataFrame.h"
#include "utils/utils.h"

/********************************************** CONSTRUCTORS ************************************************/

DataFrame::DataFrame(const std::string& csv_path) {

    // define delimiter for the file
    const char DELIM = ',';

    // create an input stream from the csv file, and throw an error if it is not possible to open the file
    std::ifstream file(csv_path);
    if (!file) throw std::runtime_error("Cannot open file: " + csv_path);

    // allocate the strings where we will temporarily store the data while we process the file
    std::string line, entry;

    // initially, we read the header (i.e. the first line of the csv) and we create a string-stream from it
    std::getline(file, line);
    std::istringstream header_stream(line);

    // we store the names of the columns. NB: we skip the first entry, i.e. the top-left corner of the table
    bool skip_first = true;
    while (std::getline(header_stream, entry, DELIM)) {
        if (skip_first) {
            skip_first = false;
            continue; // first entry is usually the name of the index column
        }
        columns.push_back(entry);
    }

    // we then read the rows. For each row: we import it,
    while (std::getline(file, line)) {
        // we create a string-stream and read all the entries (tokens) in the row
        std::istringstream row_stream(line);
        std::vector<std::string> tokens;
        while (std::getline(row_stream, entry, DELIM))
            tokens.push_back(entry);

        // sanity check (we verify that all rows are well-defined)
        if (tokens.size() != columns.size() + 1)
            throw std::runtime_error("Malformed row: " + line);

        // we store the index
        index.push_back(tokens[0]);

        // we store the content of the row
        std::vector<double> row;
        for (unsigned i = 1; i < tokens.size(); ++i)
            row.push_back(std::stod(tokens[i]));
        data.push_back(row);
    }

    index_name = "Date";

}

DataFrame::DataFrame(const std::vector<std::string>& index,
                     const std::vector<std::string>& columns)
        : index(index), columns(columns), index_name("Date")
{
    data.resize(index.size(), std::vector<double>(columns.size(), 0.0));
}


double& DataFrame::at(const std::string& row_label, const std::string& col_label) {
    auto row_it = std::find(index.begin(), index.end(), row_label);
    if (row_it == index.end())
        throw std::out_of_range("Row label not found: " + row_label);

    auto col_it = std::find(columns.begin(), columns.end(), col_label);
    if (col_it == columns.end())
        throw std::out_of_range("Column label not found: " + col_label);

    size_t row_idx = std::distance(index.begin(), row_it);
    size_t col_idx = std::distance(columns.begin(), col_it);

    return data[row_idx][col_idx];
}


const double& DataFrame::at(const std::string& row_label, const std::string& col_label) const {
    auto row_it = std::find(index.begin(), index.end(), row_label);
    if (row_it == index.end())
        throw std::out_of_range("Row label not found: " + row_label);

    auto col_it = std::find(columns.begin(), columns.end(), col_label);
    if (col_it == columns.end())
        throw std::out_of_range("Column label not found: " + col_label);

    size_t row_idx = std::distance(index.begin(), row_it);
    size_t col_idx = std::distance(columns.begin(), col_it);

    return data[row_idx][col_idx];
}


void DataFrame::filter_by_time_range(const std::string& start, const std::string& end) {
    auto lower = std::lower_bound(index.begin(), index.end(), start);
    auto upper = std::lower_bound(index.begin(), index.end(), end);
    unsigned lo = std::distance(index.begin(), lower);
    unsigned hi = std::distance(index.begin(), upper);

    index = std::vector<std::string>(index.begin() + lo, index.begin() + hi);
    data  = std::vector<std::vector<double>>(data.begin() + lo, data.begin() + hi);
}


void DataFrame::filter_by_columns(const std::vector<std::string>& selected_columns) {

    // allocate vector where we store the indices of all columns
    std::vector<unsigned> col_indices;
    col_indices.reserve(selected_columns.size());

    // store column indices
    for (const auto& name : selected_columns)
            col_indices.push_back(find_column(name));

    // extract the content of the selected columns and store in a temporary data structure
    std::vector<std::vector<double>> new_data(data.size(), std::vector<double>(col_indices.size()));
    for (unsigned i = 0; i < data.size(); ++i)
        for (unsigned j = 0; j < col_indices.size(); ++j)
            new_data[i][j] = data[i][col_indices[j]];

    // at last, we copy the content of the temporary array and we overwrite the columns
    data = std::move(new_data);
    columns = selected_columns;
}


void DataFrame::filter_by_hour(unsigned target_hour) {

    // we create temporary data-structure to save the entries while processing the dataframe
    std::vector<std::string> new_index;
    std::vector<std::vector<double>> new_data;

    for (unsigned i = 0; i < index.size(); ++i) {
        // we extract "hour" substring (i.e. the 11th and 12th characters of the "yyyy-mm-dd HH:MM:SS" string)
        //unsigned hour = static_cast<unsigned> (std::stoi(index[i].substr(11, 2)));
        unsigned hour = utils::get_hour_of_day(index[i]);
        // if hour is the same as target_hour, we copy the corresponding value
        if (hour == target_hour) {
            new_index.push_back(index[i]);
            new_data.push_back(data[i]);
        }
    }

    // finally, we replace the original vectors with the new ones
    index = std::move(new_index);
    data = std::move(new_data);
}


void DataFrame::apply_log(const std::vector<std::string>& selected_columns) {

    // Step 1: we find and we store column indices
    std::vector<unsigned> col_indices;
    col_indices.reserve(selected_columns.size());
    for (const auto& name : selected_columns)
        col_indices.push_back(find_column(name));

    // step 2: Apply log transformation
    for (auto& row : data) {
        for (unsigned j : col_indices) {
            double& val = row[j];
            if (val <= 0.0)
                throw std::runtime_error("Cannot take log of non-positive value: " + std::to_string(val));
            val = std::log(val);
        }
    }
}


unsigned DataFrame::find_column(const std::string& name) const {
    auto it = std::find(columns.begin(), columns.end(), name);
    if (it == columns.end())
        throw std::runtime_error("Column not found: " + name);
    return std::distance(columns.begin(), it);
}


void DataFrame::to_csv(const std::string& filename) const {

    // generate the output stream on the csv file
    std::ofstream file(filename);

    // set precision of decimal digits
    file << std::fixed << std::setprecision(utils::OUTPUT_DIGITS);

    // write header, with "Date" in the corner and then the column names
    file << index_name;
    for (const auto& col : columns)
        file << "," << col;
    file << "\n";

    // write rows and content of the dataset
    for (size_t i = 0; i < index.size(); ++i) {
        file << index[i];
        for (size_t j = 0; j < columns.size(); ++j)
            file << "," << data[i][j];
        file << "\n";
    }

    file.close();
}