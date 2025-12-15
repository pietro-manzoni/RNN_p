/*
Author: P.Manzoni & R.Baviera
Last Modified: P.M., 22-05-2025
Copyright (c) 2025, Politecnico di Milano, QFinLab
*/

#include <iomanip>
#include <vector>
#include <mpi.h>

#include "dataframe/DataFrame.h"
#include "models/LinearModel.h"
#include "models/Recurrent.h"
#include "preprocessing/Scaler.h"
#include "utils/utils.h"

int main(int argc, char** argv) {

    /* ============================= *
     *          MPI SETUP            *
     * ============================= */
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // load configuration from config.txt
    auto config = utils::load_config("../config.txt");

    /* ============================= *
     *        Model Settings         *
     * ============================= */

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

    /* ============================= *
     *            Seeds              *
     * ============================= */

    std::vector<unsigned> seeds = {
        101, 103, 106, 109, 117, 132, 144, 166, 173, 182,
        201, 207, 212, 236, 248, 274, 281, 294, 306, 319,
        328, 346, 358, 364, 382, 404, 421, 438, 459, 463,
        472, 488, 504, 514, 533, 567, 584, 593, 602, 612
    };

    std::vector<std::string> seed_labels(seeds.size());
    for (size_t i = 0; i < seeds.size(); ++i)
        seed_labels[i] = std::to_string(i + 1);

    /* ============================= *
     *        Local results          *
     * ============================= */

    std::vector<int>    local_ids;
    std::vector<double> local_mape, local_rmse, local_apl;

    /* ============================= *
     *   Parallel loop over seeds    *
     * ============================= */

    for (unsigned seed_num = rank; seed_num < seeds.size(); seed_num += size) {

        std::cout << "> Rank " << rank << " | Repetition # "
                  << (seed_num + 1) << "/" << seeds.size() << std::endl;

        const unsigned SEED = seeds[seed_num];

        /* =================================== *
         *  Import and normalise dataframes    *
         * =================================== */

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

        /* ================================ *
         *    Linear Model (Seasonality)    *
         * ================================ */

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
                                     const std::string &start,
                                     const std::string &end,
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
                  "../results/run" + seed_labels[seed_num] +
                  "_rank" + std::to_string(rank) + "_log.csv");

        DataFrame y_test_hat = model.predict(nn_test_X);

        /* ============================= *
         *      Assemble forecasts       *
         * ============================= */

        DataFrame mean_forecast(y_test_hat.index, {"Forecast"});

        double y_min, y_max;
        std::tie(y_min, y_max) = minmax_scaler.get_minmax("Demand");

        for (const std::string &ss : y_test_hat.index) {

            y_test_hat.at(ss, "Mu") += seasonal_OutSample.at(ss, "Seasonality");
            y_test_hat.at(ss, "Mu") =
                y_test_hat.at(ss, "Mu") * (y_max - y_min) + y_min;

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

        /* ============================= *
         *      Evaluate forecasts       *
         * ============================= */

        true_demand.filter_by_time_range(mean_forecast.index[0], END_TEST);

        double mape_test = utils::MAPE(mean_forecast, true_demand) * 100.0;
        double rmse_test = utils::RMSE(mean_forecast, true_demand);
        double apl_test  = USE_PROBABILISTIC ?
                           utils::APL(y_test_hat, true_demand) : 0.0;

        local_ids.push_back(seed_num);
        local_mape.push_back(mape_test);
        local_rmse.push_back(rmse_test);
        local_apl.push_back(apl_test);
    }

    /* ============================= *
     *        Gather results         *
     * ============================= */

    int local_n = local_ids.size();
    std::vector<int> counts(size);

    MPI_Gather(&local_n, 1, MPI_INT,
               counts.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    std::vector<int> displs;
    int total = 0;
    if (rank == 0) {
        displs.resize(size);
        for (int i = 0; i < size; ++i) {
            displs[i] = total;
            total += counts[i];
        }
    }

    std::vector<int>    all_ids(total);
    std::vector<double> all_mape(total), all_rmse(total), all_apl(total);

    MPI_Gatherv(local_ids.data(), local_n, MPI_INT,
                all_ids.data(), counts.data(), displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Gatherv(local_mape.data(), local_n, MPI_DOUBLE,
                all_mape.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Gatherv(local_rmse.data(), local_n, MPI_DOUBLE,
                all_rmse.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Gatherv(local_apl.data(), local_n, MPI_DOUBLE,
                all_apl.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ============================= *
     *        Final output           *
     * ============================= */

    if (rank == 0) {

        DataFrame results(seed_labels, {"MAPE", "RMSE", "APL"});
        results.index_name = "Run";

        for (int i = 0; i < total; ++i) {
            int s = all_ids[i];
            results.at(seed_labels[s], "MAPE") = all_mape[i];
            results.at(seed_labels[s], "RMSE") = all_rmse[i];
            results.at(seed_labels[s], "APL")  = all_apl[i];
        }

        results.to_csv("../results/performance_metrics.csv");
    }

    MPI_Finalize();
    return 0;
}


