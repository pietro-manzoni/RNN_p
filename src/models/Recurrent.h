/*
Author: P.Manzoni & R.Baviera
Last Modified: P.M., 22-05-2025
Copyright (c) 2025, Politecnico di Milano, QFinLab
*/

#ifndef RNN_p_RECURRENT_H
#define RNN_p_RECURRENT_H

#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#include "dataframe/DataFrame.h"
#include "linalg/AlgebraicOperations.h"
#include "linalg/Matrix.h"
#include "linalg/Vector.h"


class Recurrent {


// ===================================
//   Attributes
// ===================================

public:

    // structure of the network (Number of Neurons)
    const unsigned INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE;

    // structure of the network (Number of Feedbacks)
    const std::vector<unsigned int> LAGS;
    const unsigned N_LAGS;

    // length of the processed subsequences
    const unsigned WINDOW_SIZE;

    // number of weights
    const unsigned N_THETA, N_PHI;

    // weights (matrices and vectors to be trained)
    Matrix U, V;
    Vector b, c;
    std::vector<Matrix> W;

    // gradients for Adam
    Vector grad_theta, grad_phi;

    // Adam
    const double BETA1 = 0.9, BETA2 = 0.999, EPSILON_HAT = 1e-6;
    double learning_rate = 0;
    Vector g_theta_mean, g_theta_variance, g_phi_mean, g_phi_variance;

    // best weights found during optimization (validation-wise)
    Matrix U_best, V_best;
    Vector b_best, c_best;
    std::vector<Matrix> W_best;



// ===================================
//   Constructor
// ===================================

public:

    Recurrent(unsigned INPUT_NEURONS_, unsigned HIDDEN_NEURONS_, unsigned OUTPUT_NEURONS_,
              const std::vector<unsigned>& LAGS_, unsigned TAU_);



// ===================================
//   Activation Function
// ===================================

private:

    static Vector sigmoid(Vector& v);
    static Vector derivative_sigmoid(Vector& h);



// ===================================
//   Loss Functions
// ===================================

private:

    static double MSE(const Vector &y_hat, const Vector &y_true);
    static Vector derivative_MSE(const Vector &y_hat, const Vector &y_true);

    static double NLL(const Vector& y_hat, const Vector& y_true);
    static Vector derivative_NLL(const Vector& y_hat, const Vector& y_true);



// ===================================
//   Optimizer
// ===================================

private:

    void Adam(const Vector &grad_theta_new, const Vector &grad_phi_new);



// ===================================
//   Fit and Predict Methods
// ===================================

private:

    std::vector<std::vector<Vector>> data_windowing(const std::vector<std::vector<double>>& data_x) const;

    std::pair<std::vector<std::vector<Vector>>,
                          std::vector<Vector> > data_windowing(const std::vector<std::vector<double>>& data_x,
                                                               const std::vector<std::vector<double>>& data_y) const;

    Vector predict_aux(const std::vector<Vector>& subsequence_x) const;

    /*
    static void data_windowing(const std::vector<Vector>& x_timeseries, const std::vector<Vector>& y_timeseries,
                               std::vector<std::vector<Vector>>& x_windows, std::vector<Vector>& y_windows,
                               unsigned WINDOW_SIZE);
    */

public:

    void fit(const DataFrame& df_train_x, const DataFrame& df_train_y,
             const DataFrame& df_valid_x, const DataFrame& df_valid_y,
             double LEARNING_RATE, unsigned EPOCHS, unsigned BATCH_SIZE, unsigned PATIENCE,
             const std::string& filename);

    DataFrame predict(const DataFrame& x_input) const;



// ===================================
//   Early Stopping
// ===================================

private:

    void save_best_weights();
    void restore_best_weights();



// ===================================
//   Gradient Algorithms: RTRL
// ===================================

public:

    void RTRL(const std::vector<Vector>& x_input, const Vector& y_true,
              Vector& gradient_theta, Vector& gradient_phi);

private:

    void partial_derivatives_RTRL(Matrix &jacobian_theta, Matrix &jacobian_phi,
                                  const Vector &x, const Vector &h, const std::vector<Vector>& y_history,
                                  const Matrix &VA, unsigned t);



// ===================================
//   Gradient Algorithms: AAD
// ===================================

public:

    void AAD(const std::vector<Vector>& x_input, const Vector& y_true,
             Vector& gradient_theta, Vector& gradient_phi);

private:

    void partial_derivatives_AAD(Vector &gradient_theta, Vector &gradient_phi, const Vector &x, const Vector &h,
                                 const std::vector<Vector> &y_history, const Vector &z_t, const Vector &g_t,
                                 unsigned int t);

};


#endif //RNN_p_RECURRENT_H