/*
Author: P.Manzoni & R.Baviera
Last Modified: P.M., 22-05-2025
Copyright (c) 2025, Politecnico di Milano, QFinLab
*/

#include <iomanip>
#include <vector>

#include "dataframe/DataFrame.h"
#include "models/LinearModel.h"
#include "models/Recurrent.h"
#include "preprocessing/Scaler.h"
#include "utils/utils.h"


int main() {

    // load configuration from config.txt
    auto config = utils::load_config("../config.txt");

    /* ============================= *
     *        Model Settings         *
     * ============================= */

    // file where to read the dataframe from
    const auto FILENAME_IN = config.FILENAME_IN;

    // dates for the analysis (dates must be formatted as "yyyy-mm-dd HH:MM:SS")
    const auto START_TRAIN = config.START_TRAIN, END_TRAIN = config.END_TRAIN;
    const auto START_VALID = config.START_VALID, END_VALID = config.END_VALID;
    const auto START_TEST  = config.START_TEST,  END_TEST  = config.END_TEST;

    // probabilistic forecasting (true) vs point forecasting (false)
    const auto USE_PROBABILISTIC = config.USE_PROBABILISTIC;

    // RNN(p) architecture
    const auto HIDDEN_SIZE = config.HIDDEN_SIZE, TAU = config.TAU;
    const auto LAGS = config.LAGS;

    // optimization algorithm (Adam)
    const auto EPOCHS = config.EPOCHS, BATCH_SIZE = config.BATCH_SIZE, PATIENCE = config.PATIENCE;
    const auto LEARNING_RATE = config.LEARNING_RATE;

    // features for the Linear model
    const auto REGRESSORS_LM = config.REGRESSORS_LM;

    // features for the RNN(p) model
    const auto REGRESSORS_NN = config.REGRESSORS_NN;

    // set precision for log and output files
    std::cout << std::fixed << std::setprecision(utils::OUTPUT_DIGITS);

    /* ===================================== *
     *   Train models and assess forecasts   *
     * ===================================== */

    // we perform 10 repetitions of training using different random seeds
    //std::vector<unsigned> seeds = {218, 265, 313, 546, 733, 755, 789, 791, 807, 939}; // original
    /*
    std::vector<unsigned> seeds = {218, 265, 313, 546, 733, 755, 789, 791, 807, 939,
    115, 121, 128, 150, 243, 337, 379, 391, 430, 443, 450, 457, 486, 496, 542, 550, 579, 617, 623,    
    635, 675, 700, 705, 726, 778, 786, 792, 878, 938, 977};
    */

    std::vector<unsigned> seeds = {
    101, 103, 106, 109, 117, 132, 144, 166, 173, 182,
    201, 207, 212, 236, 248, 274, 281, 294, 306, 319,
    328, 346, 358, 364, 382, 404, 421, 438, 459, 463,
    472, 488, 504, 514, 533, 567, 584, 593, 602, 612
    };

    // prepare the labels for the dataframe where we store the output
    std::vector<std::string> seed_labels(seeds.size());
    for (size_t i = 0; i < seeds.size(); ++i)
        seed_labels[i] = std::to_string(i+1);

    // initialize the dataframe where we store the output
    DataFrame results(seed_labels, {"MAPE", "RMSE", "APL"});
    results.index_name = "Run";

    // loop over the seeds
    for (unsigned seed_num = 0; seed_num < seeds.size(); ++seed_num) {

        // print for user interface and initialize the seed of the random number generator
        std::cout << "> Repetition # " << (seed_num + 1) << "/" << seeds.size() << std::endl;
        const unsigned SEED = seeds[seed_num];


        /* =================================== *
         *  Import and normalise dataframes    *
         * =================================== */

        // we import the raw dataframe from the csv file
        DataFrame raw_df(FILENAME_IN);

        // we build the In-Sample (from the raw dataframe) and the Out-of-Sample dataframe (a copy of the former)
        DataFrame df_InSample = utils::build_dataset(raw_df);
        //DataFrame df_InSample("../data/GEFCom_df.csv");
        DataFrame df_OutSample(df_InSample);

        // moreover, we store the true demand for future use, before performing any filtering or scaling of the dataframes
        DataFrame true_demand(df_OutSample);
        true_demand.filter_by_columns({"Demand"});

        // we restrict the datasets to the dates in the pre-selected range for In-Sample and Out-of-Sample, respectively
        df_InSample.filter_by_time_range(START_TRAIN, END_VALID);
        df_OutSample.filter_by_time_range(START_TEST, END_TEST);

        // we apply logarithm to the "Demand" to obtain the log-demand
        df_InSample.apply_log({"Demand"});
        df_OutSample.apply_log({"Demand"});

        // we initialize MinMaxScaler, we fit it on In-Sample data ...
        Scaler minmax_scaler;
        minmax_scaler.fit(df_InSample);
        // ... and we MinMax-transform both In-Sample and Out-of-Sample data
        minmax_scaler.transform(df_InSample);
        minmax_scaler.transform(df_OutSample);


        /* ================================ *
         *    Linear Model (Seasonality)    *
         * ================================ */

        // we instantiate null dataframes where we collect the forecasts of the seasonal part (In-Sample and Out-of-Sample)
        DataFrame seasonal_InSample(df_InSample.index, {"Seasonality"});
        DataFrame seasonal_OutSample(df_OutSample.index, {"Seasonality"});

        // we capture the seasonal part via OLS (hourly-wise)
        for (unsigned hh = 0; hh < 24; ++hh) {

            // (i) we create the In-Sample regressors at for the hour hh
            DataFrame tmp_InSample_X(df_InSample);
            tmp_InSample_X.filter_by_columns(REGRESSORS_LM);
            tmp_InSample_X.filter_by_hour(hh);

            // (ii) we create the In-Sample target at for the hour hh
            DataFrame tmp_InSample_Y(df_InSample);
            tmp_InSample_Y.filter_by_columns({"Demand"});
            tmp_InSample_Y.filter_by_hour(hh);

            // (iii) we create the Out-of_Sample regressors at for the hour hh
            DataFrame tmp_OutSample_X(df_OutSample);
            tmp_OutSample_X.filter_by_columns(REGRESSORS_LM);
            tmp_OutSample_X.filter_by_hour(hh);

            // (iv) we create the Out-of_Sample regressors at for the hour hh
            DataFrame tmp_OutSample_Y(df_OutSample);
            tmp_OutSample_Y.filter_by_columns({"Demand"});
            tmp_OutSample_Y.filter_by_hour(hh);

            // (v) we fit the linear model on In-Sample data via OLS
            LinearModel lm(tmp_InSample_X.data, tmp_InSample_Y.data);

            // (vi) we compute fitted values and predicted values, and store them
            std::vector<double> lm_forecast_InSample = lm.predict(tmp_InSample_X.data);
            std::vector<double> lm_forecast_OutSample = lm.predict(tmp_OutSample_X.data);

            // (vii) we fill in the dataframe of In-Sample forecasts (i.e. the fitted values), declared outside the loop
            for (unsigned i = 0; i < tmp_InSample_X.index.size(); ++i)
                seasonal_InSample.at(tmp_InSample_X.index[i], "Seasonality") = lm_forecast_InSample[i];

            // (viii) similarly, we fill in the dataframe of Out-of-Sample forecasts (i.e. the predicted values)
            for (unsigned i = 0; i < tmp_OutSample_X.index.size(); ++i)
                seasonal_OutSample.at(tmp_OutSample_X.index[i], "Seasonality") = lm_forecast_OutSample[i];

        }


        /* ============================= *
         *         RNN(p) Model          *
         * ============================= */

        // we create the train/validation/test split that will be processed by the NN. To do so in a clean way,
        // we create a helper function (lambda function) that performs the filtering
        auto prepare_nn_dataset = [](DataFrame &X, DataFrame &Y,
                                     const std::string &start, const std::string &end,
                                     const std::vector<std::string> &nn_regressors,
                                     const DataFrame &seasonal_dataframe) -> void {

            // apply time filter
            X.filter_by_time_range(start, end);
            Y.filter_by_time_range(start, end);

            // filter by selected columns
            X.filter_by_columns(nn_regressors);
            Y.filter_by_columns({"Demand"});

            // remove the seasonal component from the target variable
            for (const std::string &date: Y.index)
                Y.at(date, "Demand") -= seasonal_dataframe.at(date, "Seasonality");

        };

        // (i) construct the training set
        DataFrame nn_train_X(df_InSample), nn_train_Y(df_InSample);
        prepare_nn_dataset(nn_train_X, nn_train_Y, START_TRAIN, END_TRAIN, REGRESSORS_NN, seasonal_InSample);
        // (ii) construct the validation set
        DataFrame nn_valid_X(df_InSample), nn_valid_Y(df_InSample);
        prepare_nn_dataset(nn_valid_X, nn_valid_Y, START_VALID, END_VALID, REGRESSORS_NN, seasonal_InSample);
        // (iii) construct the test set
        DataFrame nn_test_X(df_OutSample), nn_test_Y(df_OutSample);
        prepare_nn_dataset(nn_test_X, nn_test_Y, START_TEST, END_TEST, REGRESSORS_NN, seasonal_OutSample);

        // we instantiate the RNN(p) Model
        utils::rng.seed(SEED);
        const unsigned INPUT_SIZE = REGRESSORS_NN.size(), OUTPUT_SIZE = USE_PROBABILISTIC ? 2 : 1;
        Recurrent model(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LAGS, TAU);

        // we fit the RNN(p) Model
        utils::rng.seed(SEED);
        model.fit(nn_train_X, nn_train_Y, nn_valid_X, nn_valid_Y, LEARNING_RATE, EPOCHS, BATCH_SIZE, PATIENCE,
                  "../results/run" + seed_labels[seed_num] + "_log.csv");

        // we predict the values on the test set using the RNN(p) Model
        DataFrame y_test_hat = model.predict(nn_test_X);


        /* ============================= *
         *      Assemble forecasts       *
         * ============================= */

        // we instantiate the collector for the mean forecast
        DataFrame mean_forecast(y_test_hat.index, {"Forecast"});

        // we prepare the min and the max stored in the scaler
        double y_min = 0., y_max = 0.;
        std::tie(y_min, y_max) = minmax_scaler.get_minmax("Demand");

        // in the point forecasting case
        if (!USE_PROBABILISTIC) {
            for (const std::string &ss: y_test_hat.index) {
                // i) we add the seasonal forecast
                y_test_hat.at(ss, "Mu") += seasonal_OutSample.at(ss, "Seasonality");
                // ii) we unscale the point forecast
                y_test_hat.at(ss, "Mu") = y_test_hat.at(ss, "Mu") * (y_max - y_min) + y_min;
                // iii) we compute the actual forecast for the demand by exponentiation
                mean_forecast.at(ss, "Forecast") = exp(y_test_hat.at(ss, "Mu"));
            }
        }
            // in the probabilistic forecasting case
        else {
            for (const std::string &ss: y_test_hat.index) {
                // i) we add the seasonal forecast
                y_test_hat.at(ss, "Mu") += seasonal_OutSample.at(ss, "Seasonality");
                // ii) we unscale the forecasts for the mean and for the standard devation
                y_test_hat.at(ss, "Mu") = y_test_hat.at(ss, "Mu") * (y_max - y_min) + y_min;
                y_test_hat.at(ss, "Sigma") = y_test_hat.at(ss, "Sigma") * (y_max - y_min);
                // iii) we compute the actual forecast as the mean of the forecasted lognormal distribution
                mean_forecast.at(ss, "Forecast") = exp(y_test_hat.at(ss, "Mu") +
                                                       0.5 * y_test_hat.at(ss, "Sigma") * y_test_hat.at(ss, "Sigma"));
            }
        }

        // in both cases, we save the forecasts for the log-Demand to a csv file
        y_test_hat.to_csv("../results/run" + seed_labels[seed_num] + "_forecasts.csv");


        /* ============================= *
         *      Evaluate forecasts       *
         * ============================= */

        // we make sure that the dates for the true demand (stored at the beginning) match those of the forecasts
        true_demand.filter_by_time_range(mean_forecast.index[0], END_TEST);

        // we compute mape (for the forecast of the mean)
        double mape_test = utils::MAPE(mean_forecast, true_demand) * 100;
        results.at(seed_labels[seed_num], "MAPE") = mape_test;
        std::cout << "MAPE: " << mape_test << std::endl;

        // we compute rmse (for the forecast of the mean)
        double rmse_test = utils::RMSE(mean_forecast, true_demand);
        results.at(seed_labels[seed_num], "RMSE") = rmse_test;
        std::cout << "RSME: " << rmse_test << std::endl;

        // in the probabilistic case, we also compute the APL (i.e. the Average Pinball Loss)
        if (USE_PROBABILISTIC) {
            double apl_test = utils::APL(y_test_hat, true_demand);
            results.at(seed_labels[seed_num], "APL") = apl_test;
            std::cout << "APL: " << apl_test << std::endl;
        }
        else
            results.at(seed_labels[seed_num], "APL") = 0.;

    }

    // finally, we export the results to csv
    results.to_csv("../results/performance_metrics.csv");

    return 0;

} // end of main
