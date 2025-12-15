/*
Author: R.Baviera & P.Manzoni
Last Modified: P.M., 22-05-2025
Copyright (c) 2025, Politecnico di Milano, QFinLab
*/

#include <random>
#include <sstream>
#include <functional>
#include <cmath>
#include <limits>
#include <iomanip>
#include <ctime>

#include "utils.h"


namespace utils {

    // define the random number generator
    std::mt19937 rng;

    int OUTPUT_DIGITS = 12;

    // we create an anonymous namespace, which can be accessed only by the functions within this file
    namespace {

        double normpdf(double x){
            return 1 / sqrt(2*M_PI) * exp(-0.5*x*x);
        }

        double normcdf(double x){
            return 0.5 * erfc(-x * sqrt(0.5));
        }


        double norminv(double alpha){

            const double TOL = 1e-10;
            const unsigned MAX_ITER = 100;

            // consistency check
            if (alpha == 0.)
                return - std::numeric_limits<double>::infinity();
            else if (alpha == 1.)
                return + std::numeric_limits<double>::infinity();
            else if(alpha < 0. || alpha > 1.){
                std::cerr << "Warning: Invalid alpha argument provided to norminv" << std::endl;
                return std::numeric_limits<double>::quiet_NaN();
            }

            // if alpha is ok: we use Polya (1945) approximation to find initial point
            // (cf. also "Approximations to Standard Normal Distribution Function", R. Yerukala and N.K. Boiroju 2015)
            double x0 = (alpha >= 0.5 ? +1 : -1) * std::sqrt(-M_PI / 2 * std::log(4 * alpha * (1 - alpha)));

            // compute residual value
            double resid = normcdf(x0) - alpha;

            // and use Newton-Rapson for inverting the normcdf
            unsigned n_iter = 0;
            while (fabs(resid) > TOL && n_iter < MAX_ITER ){
                x0 -= resid / normpdf(x0);    //point update
                resid = normcdf(x0) - alpha;  //update residual value
                ++n_iter;                     //increase counter for max_iter criterion
            }

            // if necessary, print warning
            if (n_iter == MAX_ITER)
                std::cerr << "Warning: maximum number of iteration reached in norminv";

            return x0;

        }

        // Helper function to trim whitespace from a string
        std::string trim(const std::string& str) {
            if (str.empty()) return "";

            size_t start = str.find_first_not_of(" \t\n\r");
            size_t end = str.find_last_not_of(" \t\n\r");
            return str.substr(start, end - start + 1);
        }

        // Helper function to split a comma-separated string into a vector
        std::vector<std::string> split_list_string(const std::string& s) {
            std::vector<std::string> result;
            std::stringstream ss(s);
            std::string item;
            while (std::getline(ss, item, ',')) {
                result.push_back(trim(item));
            }
            return result;
        }

        // Helper function to split a comma-separated string into a vector
        std::vector<unsigned> split_list_unsigned(const std::string& s) {
            std::vector<unsigned> result;
            std::stringstream ss(s);
            std::string item;
            while (std::getline(ss, item, ',')) {
                result.push_back(static_cast<unsigned>(std::stoul(item)));
            }
            return result;
        }


    } // end of the anonymous namespace



    void shuffle(std::vector<unsigned>& vec, std::mt19937& ran) {
        size_t n = vec.size();
        for (size_t i = n - 1; i > 0; --i) {
            size_t j = ran() % (i + 1);// Generate random index in the range [0, i]
            std::swap(vec[i], vec[j]);  // Swap the elements
        }
    }

    // Function to load the configuration from a file
    Config load_config(const std::string& filename) {
        std::ifstream file(filename);
        Config config;
        std::string line;

        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;

            size_t pos = line.find('=');
            if (pos == std::string::npos) continue;

            std::string key = trim(line.substr(0, pos));
            std::string value = trim(line.substr(pos + 1));

            if (key == "FILENAME_IN") {
                config.FILENAME_IN = value;
            } else if (key == "START_TRAIN") {
                config.START_TRAIN = value;
            } else if (key == "END_TRAIN") {
                config.END_TRAIN = value;
            } else if (key == "START_VALID") {
                config.START_VALID = value;
            } else if (key == "END_VALID") {
                config.END_VALID = value;
            } else if (key == "START_TEST") {
                config.START_TEST = value;
            } else if (key == "END_TEST") {
                config.END_TEST = value;
            } else if (key == "USE_PROBABILISTIC") {
                config.USE_PROBABILISTIC = (value == "true");
            } else if (key == "HIDDEN_SIZE") {
                config.HIDDEN_SIZE = std::stoi(value);
            } else if (key == "LAGS") {
                config.LAGS = split_list_unsigned(value);
            } else if (key == "EPOCHS") {
                config.EPOCHS = std::stoi(value);
            } else if (key == "BATCH_SIZE") {
                config.BATCH_SIZE = std::stoi(value);
            } else if (key == "PATIENCE") {
                config.PATIENCE = std::stoi(value);
            } else if (key == "LEARNING_RATE") {
                config.LEARNING_RATE = std::stod(value);
            } else if (key == "TAU") {
                config.TAU = std::stoi(value);
            } else if (key == "REGRESSORS_LM") {
                config.REGRESSORS_LM = split_list_string(value);
            } else if (key == "REGRESSORS_NN") {
                config.REGRESSORS_NN = split_list_string(value);
            }
        }

        return config;
    }

    double RMSE(const DataFrame& forecast, const DataFrame& realized) {

        // sanity check: number of rows must match
        if (realized.data.size() != forecast.data.size())
            throw std::invalid_argument("Realized and forecast data must have the same number of rows.");

        // instantiate the relevant variables
        double mse = 0.0;
        const unsigned N = realized.data.size();

        // compute Mean Squared Error
        for (size_t i = 0; i < N; ++i) {
            double error = forecast.data[i][0] - realized.data[i][0];
            mse += (error * error) / N;
        }

        return std::sqrt(mse);
    }


    double MAPE(const DataFrame& forecast, const DataFrame& realized) {

        // sanity check: number of rows must match
        if (realized.data.size() != forecast.data.size())
            throw std::invalid_argument("Realized and forecast data must have the same number of rows.");

        // instantiate the relevant variables
        double mape = 0.0;
        const unsigned N = realized.data.size();

        // compute the Mean Absolute Percentage Error
        for (size_t i = 0; i < N; ++i) {
            double relative_error = std::abs(forecast.data[i][0] / realized.data[i][0] - 1);
            mape += relative_error / N;
        }

        return mape;
    }


    double APL(const DataFrame& forecast, const DataFrame& realized) {

        // sanity checks
        if (forecast.data.size() != realized.data.size())
            throw std::invalid_argument("Forecast and realized must have the same number of rows.");

        // size of the forecasts/realised values
        const unsigned N = forecast.data.size();

        // average pinball loss
        double apl = 0.0;

        // we consider all quantiles from 1% to 99%
        std::vector<double> quantile_levels;
        for (int i = 1; i < 100; ++i)
            quantile_levels.push_back(i / 100.0);

        // we loop over the considered percentiles
        for (auto q : quantile_levels) {

            // pinball loss at level q
            double loss_q = 0.0;

            // inverse of normal-cdf at level q
            double z = norminv(q);

            // we loop over the forecasts
            for (unsigned i = 0; i < N; ++i) {

                // extract mu and sigma for each forecast, and the realised demand
                double mu = forecast.data[i][0], sigma = forecast.data[i][1];
                double y0 = realized.data[i][0];

                // compute the q-quantile
                double quantile = std::exp(mu + z * sigma);

                // compute pinball loss for the i-th forecast
                double loss = (1 - q) * (quantile - y0) * (quantile >= y0) +
                                   q  * (y0 - quantile) * (quantile <  y0);

                // add the contribution of the i-th forecast
                loss_q += loss / static_cast<double>(N);

            }

            // add the contribution for the quantile q
            apl += loss_q;

        }

        // return average pinball loss across all quantiles
        return apl / static_cast<double>(quantile_levels.size());
    }

    unsigned get_day_of_year(const std::string& datetime_str) {
        std::tm tm{};
        // take only the "YYYY-MM-DD", as C++ has notorious issues in managing the daylight saving time
        std::istringstream ss(datetime_str.substr(0, 10));
        ss >> std::get_time(&tm, "%Y-%m-%d");
        mktime(&tm);
        return tm.tm_yday; // 01-Jan = 0
    }

    unsigned get_day_of_week(const std::string& datetime_str) {
        std::tm tm{};
        // take only the "YYYY-MM-DD", as C++ has notorious issues in managing the daylight saving time
        std::istringstream ss(datetime_str.substr(0, 10));
        ss >> std::get_time(&tm, "%Y-%m-%d");
        mktime(&tm);

        // Shift so that Monday = 0, ..., Sunday = 6
        return (tm.tm_wday + 6) % 7;
    }

    unsigned get_hour_of_day(const std::string& datetime_str) {
        return static_cast<unsigned int>(std::stoul(datetime_str.substr(11, 2)));
    }

    // Function to extract the year from a date string
    unsigned get_year(const std::string& datetime_str) {
        return static_cast<unsigned>(std::stoul(datetime_str.substr(0, 4)));
    }


    DataFrame build_dataset(const DataFrame& raw_df){

        // list the new columns, i.e. those we add in this function
        std::vector<std::string> all_columns = {"Trend",
                                                "SY1", "CY1", "SY2", "CY2",
                                                "SD1", "CD1", "SD2", "CD2",
                                                "DoW_0", "DoW_1", "DoW_2", "DoW_3", "DoW_4", "DoW_5", "DoW_6"};

        // add the existing columns in front
        all_columns.insert(all_columns.begin(), raw_df.columns.begin(), raw_df.columns.end());

        // create an empty dataframe with the same rows as the raw one and an extended number of columns
        DataFrame df(raw_df.index, all_columns);

        // copy the content of the raw dataframe in the new one (where the columns exist)
        for (unsigned i = 0; i < raw_df.index.size(); ++i)
            for (unsigned j = 0; j < raw_df.columns.size(); ++j)
                df.data[i][j] = raw_df.data[i][j];

        // number of columns in the raw dataframe
        const unsigned C = raw_df.columns.size();

        // fill the new DataFrame with the data from the raw DataFrame and additional computed values
        for (unsigned i = 0; i < raw_df.index.size(); ++i) {

            // Add Trend (sequence from 0 to n)
            df.data[i][C+0] = i; // Trend column

            // Add Fourier terms for day of the year
            unsigned day_of_year_value = get_day_of_year(raw_df.index[i]);
            unsigned year = get_year(raw_df.index[i]);
            unsigned denominator = (year % 4 == 0) ? 366 : 365;

            df.data[i][C+1] = std::sin(2 * M_PI * day_of_year_value / denominator); // SY1
            df.data[i][C+2] = std::cos(2 * M_PI * day_of_year_value / denominator); // CY1
            df.data[i][C+3] = std::sin(4 * M_PI * day_of_year_value / denominator); // SY2
            df.data[i][C+4] = std::cos(4 * M_PI * day_of_year_value / denominator); // CY2

            // Add Fourier terms for hour of the day
            unsigned hour_value = get_hour_of_day(raw_df.index[i]);
            df.data[i][C+5] = std::sin(2 * M_PI * hour_value / 24); // SD1
            df.data[i][C+6] = std::cos(2 * M_PI * hour_value / 24); // CD1
            df.data[i][C+7] = std::sin(4 * M_PI * hour_value / 24); // SD2
            df.data[i][C+8] = std::cos(4 * M_PI * hour_value / 24); // CD2

            // Day of week (One-Hot Encoding)
            unsigned day_of_week = get_day_of_week(raw_df.index[i]);  // Simple placeholder for day of week
            for (int j = 0; j < 7; ++j)
                df.data[i][C+9+j] = (j == day_of_week) ? 1 : 0; // One-hot encoding
        }

        return df;

    }

} // end of utils
