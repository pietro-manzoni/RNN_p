/*
Author: P.Manzoni & R.Baviera
Adapted for GNU Parallel
Last Modified: 15-12-2025
Copyright (c) 2025, Politecnico di Milano, QFinLab
*/

#include <iomanip>
#include <vector>
#include <string>
#include <iostream>

#include "dataframe/DataFrame.h"
#include "models/LinearModel.h"
#include "models/Recurrent.h"
#include "preprocessing/Scaler.h"
#include "utils/utils.h"

int main(int argc, char** argv) {

    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <seed>" << std::endl;
        return 1;
    }

    unsigned SEED = std::stoi(argv[1]);
    std::string seed_label = std::to_string(SEED);

    // load configuration
    auto config = utils::load_config("../config.txt");

    const auto FILENAME_IN = config.FILENAME_IN;
    const auto START_TRAIN = config.START_TRAIN, END_TRAIN = config.END_TRAIN;
    const auto START_VALID = config.START_VALID, END_VALID = config.END_VALID;
    const auto START_TEST  = config.START_TEST,  END_TEST  = config.END_TEST;
    const auto USE_PROBABILISTIC = config.USE_PROBABILISTIC;
    const auto HIDDEN_SIZE = config.HIDDEN_SIZE, TAU = config.TAU;
    const auto LAGS = config.LAGS;
    const auto EPOCHS = config.EPOCHS, BATCH_SIZE = config.BATCH_SIZE, PATIENCE = config.PATIENCE;
    const auto LEARNING_RATE = config.LEARNING_RATE;
    const auto REGRESSORS_LM = config.REGRESSORS_LM;
    const auto REGRESSORS_NN = config.REGRESSORS_NN;

    std::cout << std::fixed << std::setprecision(utils::OUTPUT_DIGITS);

    std::cout << "> Running seed: " << SEED << std::endl;

    /* ============================= *
     *  Import and preprocess data   *
     * ============================= */

    DataFrame raw_df(FILENAME_IN);
    DataFrame df_InSample = utils::build_dataset(raw_df);
    DataFrame df_OutSample(df_InSample);

    DataFrame true_demand(df_OutSample);
    true_demand.filter_by_columns({"Demand"});

    df_InSample.filter_by_time_range(START_TRAIN, END_VALID);
    df_OutSample.filter_by_time_range(START_TEST, END_TEST);

    df_InSample.apply_log({"Demand"});
    df_OutSample.apply_log({"Demand"});

    Scaler minmax_scaler;
    minmax_scaler.fit(df_InSample);
    minmax_scaler.transform(df_InSample);
    minmax_scaler.transform(df_OutSample);

    /* ============================= *
     *    Linear Model (Seasonality) *
     * ============================= */

    DataFrame seasonal_InSample(df_InSample.index, {"Seasonality"});
    DataFrame seasonal_OutSample(df_OutSample.index, {"Seasonality"});

    for (unsigned hh = 0; hh < 24; ++hh) {

        DataFrame tmp_InSample_X(df_InSample);
        tmp_InSample_X.filter_by_columns(REGRESSORS_LM);
        tmp_InSample_X.filter_by_hour(hh);

        DataFrame tmp_InSample_Y(df_InSample);
        tmp_InSample_Y.filter_by_columns({"Demand"});
        tmp_InSample_Y.filter_by_hour(hh);

        DataFrame tmp_OutSample_X(df_OutSample);
        tmp_OutSample_X.filter_by_columns(REGRESSORS_LM);
        tmp_OutSample_X.filter_by_hour(hh);

        LinearModel lm(tmp_InSample_X.data, tmp_InSample_Y.data);

        auto lm_forecast_InSample  = lm.predict(tmp_InSample_X.data);
        auto lm_forecast_OutSample = lm.predict(tmp_OutSample_X.data);

        for (unsigned i = 0; i < tmp_InSample_X.index.size(); ++i)
            seasonal_InSample.at(tmp_InSample_X.index[i], "Seasonality") =
                lm_forecast_InSample[i];

        for (unsigned i = 0; i < tmp_OutSample_X.index.size(); ++i)
            seasonal_OutSample.at(tmp_OutSample_X.index[i], "Seasonality") =
                lm_forecast_OutSample[i];
    }

    /* ============================= *
     *         RNN(p) Model          *
     * ============================= */

    auto prepare_nn_dataset = [](DataFrame &X, DataFrame &Y,
                                 const std::string &start, const std::string &end,
                                 const std::vector<std::string> &nn_regressors,
                                 const DataFrame &seasonal_dataframe) {

        X.filter_by_time_range(start, end);
        Y.filter_by_time_range(start, end);
        X.filter_by_columns(nn_regressors);
        Y.filter_by_columns({"Demand"});

        for (const std::string &date : Y.index)
            Y.at(date, "Demand") -= seasonal_dataframe.at(date, "Seasonality");
    };

    DataFrame nn_train_X(df_InSample), nn_train_Y(df_InSample);
    prepare_nn_dataset(nn_train_X, nn_train_Y,
                       START_TRAIN, END_TRAIN,
                       REGRESSORS_NN, seasonal_InSample);

    DataFrame nn_valid_X(df_InSample), nn_valid_Y(df_InSample);
    prepare_nn_dataset(nn_valid_X, nn_valid_Y,
                       START_VALID, END_VALID,
                       REGRESSORS_NN, seasonal_InSample);

    DataFrame nn_test_X(df_OutSample), nn_test_Y(df_OutSample);
    prepare_nn_dataset(nn_test_X, nn_test_Y,
                       START_TEST, END_TEST,
                       REGRESSORS_NN, seasonal_OutSample);

    utils::rng.seed(SEED);
    Recurrent model(REGRESSORS_NN.size(),
                    HIDDEN_SIZE,
                    USE_PROBABILISTIC ? 2 : 1,
                    LAGS, TAU);

    model.fit(nn_train_X, nn_train_Y,
              nn_valid_X, nn_valid_Y,
              LEARNING_RATE, EPOCHS,
              BATCH_SIZE, PATIENCE,
              "../results/run" + seed_label + "_log.csv");

    DataFrame y_test_hat = model.predict(nn_test_X);

    /* ============================= *
     *      Assemble forecasts       *
     * ============================= */

    DataFrame mean_forecast(y_test_hat.index, {"Forecast"});

    double y_min, y_max;
    std::tie(y_min, y_max) = minmax_scaler.get_minmax("Demand");

    for (const std::string &ss : y_test_hat.index) {

        y_test_hat.at(ss, "Mu") += seasonal_OutSample.at(ss, "Seasonality");
        y_test_hat.at(ss, "Mu") = y_test_hat.at(ss, "Mu") * (y_max - y_min) + y_min;

        if (USE_PROBABILISTIC) {
            y_test_hat.at(ss, "Sigma") *= (y_max - y_min);
            mean_forecast.at(ss, "Forecast") =
                exp(y_test_hat.at(ss, "Mu") +
                    0.5 * y_test_hat.at(ss, "Sigma") *
                          y_test_hat.at(ss, "Sigma"));
        } else {
            mean_forecast.at(ss, "Forecast") =
                exp(y_test_hat.at(ss, "Mu"));
        }
    }

    y_test_hat.to_csv("../results/run" + seed_label + "_forecasts.csv");

    /* ============================= *
     *      Evaluate forecasts       *
     * ============================= */

    true_demand.filter_by_time_range(mean_forecast.index[0], END_TEST);

    double mape_test = utils::MAPE(mean_forecast, true_demand) * 100.0;
    double rmse_test = utils::RMSE(mean_forecast, true_demand);
    double apl_test  = USE_PROBABILISTIC ? utils::APL(y_test_hat, true_demand) : 0.0;

    DataFrame results({seed_label}, {"MAPE", "RMSE", "APL"});
    results.index_name = "Run";
    results.at(seed_label, "MAPE") = mape_test;
    results.at(seed_label, "RMSE") = rmse_test;
    results.at(seed_label, "APL")  = apl_test;
    results.to_csv("../results/performance_metrics_" + seed_label + ".csv");

    std::cout << "> Seed " << SEED << " finished. MAPE: " << mape_test
              << " RMSE: " << rmse_test << " APL: " << apl_test << std::endl;

    return 0;
}

