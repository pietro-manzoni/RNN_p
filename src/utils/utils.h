/*
Author: R.Baviera & P.Manzoni
Last Modified: P.M., 22-05-2025
Copyright (c) 2025, Politecnico di Milano, QFinLab
*/

#ifndef RNN_p_UTILS_H
#define RNN_p_UTILS_H

#include <random>
#include <iostream>

#include "dataframe/DataFrame.h"
#include "linalg/Vector.h"


namespace utils
{

    // Random number generator, which is the same for the whole project and shared between the functions
    extern std::mt19937 rng;

    // Decimal digits to use in terminal and output files
    extern int OUTPUT_DIGITS;

    void shuffle(std::vector<unsigned>& vec, std::mt19937& ran);

    // Function to load configuration from file
    struct Config;
    Config load_config(const std::string& filename);

    DataFrame build_dataset(const DataFrame& raw_df);

    unsigned get_day_of_year(const std::string& datetime_str);
    unsigned get_day_of_week(const std::string& datetime_str);
    unsigned get_hour_of_day(const std::string& datetime_str);
    unsigned get_year(const std::string& datetime_str);

    double RMSE(const DataFrame& forecast, const DataFrame& realized);
    double MAPE(const DataFrame& forecast, const DataFrame& realized);
    double APL(const DataFrame& forecast, const DataFrame& realized);


    // Define in detail the Config struct
    struct Config {

        // --- File Settings ---
        std::string FILENAME_IN;

        // --- Date Ranges ---
        std::string START_TRAIN, END_TRAIN;
        std::string START_VALID, END_VALID;
        std::string START_TEST, END_TEST;

        // --- Forecasting Mode ---
        bool USE_PROBABILISTIC;

        // --- RNN(p) Architecture ---
        unsigned HIDDEN_SIZE;
        std::vector<unsigned> LAGS;

        // --- Optimization Parameters ---
        unsigned EPOCHS;
        unsigned BATCH_SIZE;
        unsigned PATIENCE;
        double LEARNING_RATE;

        // --- Data Windowing ---
        unsigned TAU;

        // --- Regressors for Linear Model ---
        std::vector<std::string> REGRESSORS_LM;

        // --- Regressors for RNN Model ---
        std::vector<std::string> REGRESSORS_NN;
    };

}

#endif //RNN_p_UTILS_H
