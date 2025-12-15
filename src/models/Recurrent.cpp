/*
Author: P.Manzoni & R.Baviera
Last Modified: P.M., 22-05-2025
Copyright (c) 2025, Politecnico di Milano, QFinLab
*/

#include <cmath>
#include <iomanip>
#include <fstream>

#include "Recurrent.h"
#include "utils/utils.h"

Recurrent::Recurrent(unsigned INPUT_SIZE_, unsigned HIDDEN_SIZE_, unsigned OUTPUT_SIZE_,
                     const std::vector<unsigned int> &LAGS_, unsigned TAU_) :
                     INPUT_SIZE(INPUT_SIZE_),
                     HIDDEN_SIZE(HIDDEN_SIZE_),
                     OUTPUT_SIZE(OUTPUT_SIZE_),
                     LAGS(LAGS_),
                     N_LAGS(static_cast<unsigned>(LAGS_.size())),
                     WINDOW_SIZE(TAU_),
                     N_THETA(HIDDEN_SIZE_ * (INPUT_SIZE_ + OUTPUT_SIZE_ * N_LAGS + 1)),
                     N_PHI(OUTPUT_SIZE_ * (HIDDEN_SIZE + 1)),

                     // initialization of the vectors where we store the gradients needed for training
                     grad_theta(Vector(N_THETA, 0.)),
                     grad_phi(Vector(N_PHI, 0.)),

                     // initialization of the vectors used by Adam to update the trainable weights
                     g_theta_mean(Vector(N_THETA, 0.)),
                     g_theta_variance(Vector(N_THETA, 0.)),
                     g_phi_mean(Vector(N_PHI, 0.)),
                     g_phi_variance(Vector(N_PHI, 0.)),

                     // initialization to zero of the matrices and the vectors of trainable weights
                     U(Matrix(HIDDEN_SIZE_, INPUT_SIZE_, 0.)),
                     V(Matrix(OUTPUT_SIZE_, HIDDEN_SIZE_, 0.)),
                     b(Vector(HIDDEN_SIZE_, 0.)),
                     c(Vector(OUTPUT_SIZE_, 0.)),

                    // initialization of the matrices and the vectors that obtain the best validation performance
                     U_best(Matrix(HIDDEN_SIZE_, INPUT_SIZE_, 0.)),
                     V_best(Matrix(OUTPUT_SIZE_, HIDDEN_SIZE_, 0.)),
                     b_best(Vector(HIDDEN_SIZE_, 0.)),
                     c_best(Vector(OUTPUT_SIZE_, 0.))
                     {
                        // to conclude, we randomly initialize the trainable kernels (bias are initialized to zero)
                         std::uniform_real_distribution<double> dist(-0.05, 0.05);

                         for (unsigned i = 0; i < U.ROWS; ++i)
                             for (unsigned j = 0; j < U.COLS; ++j)
                                 U(i,j) = dist(utils::rng);

                         for (unsigned i = 0; i < V.ROWS; ++i)
                             for (unsigned j = 0; j < V.COLS; ++j)
                                 V(i,j) = dist(utils::rng);

                         for (unsigned k = 0; k < N_LAGS; ++k) {
                             // instantiate and randomly fill the k-th autoregressive kernel
                             Matrix tmp = Matrix(HIDDEN_SIZE_, OUTPUT_SIZE_, 0.);
                             for (unsigned i = 0; i < tmp.ROWS; ++i)
                                 for (unsigned j = 0; j < tmp.COLS; ++j)
                                     tmp(i,j) = dist(utils::rng);

                            // copy the created matrix into the actual member W of the class
                             W.push_back(tmp);

                             // also, fill with zeros the matrices that obtain the best validation performance
                             W_best.emplace_back(HIDDEN_SIZE_, OUTPUT_SIZE_, 0.);
                         }

                     }



// ===================================
// Activation Function
// ===================================

Vector Recurrent::sigmoid(Vector &v) {

    Vector out(v.SIZE, 0);
    for (unsigned i = 0; i < v.SIZE; ++i)
        out(i) = 1 / (1 + std::exp(-v(i)));
    return out;

}

Vector Recurrent::derivative_sigmoid(Vector &h) {

    Vector out(h.SIZE, 0);
    for (unsigned i = 0; i < h.SIZE; ++i)
        out(i) = h(i) * (1-h(i));
    return out;

}


// ===================================
// Loss Functions
// ===================================

double Recurrent::MSE(const Vector& y_hat, const Vector& y_true) {

    return (y_hat(0) - y_true(0)) * (y_hat(0) - y_true(0));

}

Vector Recurrent::derivative_MSE(const Vector& y_hat, const Vector& y_true) {

    const double DERIVATIVE = 2 * (y_hat(0) - y_true(0));
    return Vector(1, DERIVATIVE);

}


double Recurrent::NLL(const Vector& y_hat, const Vector& y_true) {

    const double SIGMA_ROBUST = std::abs(y_hat(1)) + 1e-9; // we add a negligible value to sigma for robustness
    const double Z_SCORE = (y_true(0) - y_hat(0)) / SIGMA_ROBUST;

    // compute Negative (Gaussian) Log-Likelihood
    // return .5 * log(2 * M_PI) + .5 * log( y_hat(1) * y_hat(1) ) + .5 * Z_SCORE * Z_SCORE;
    return .5 * log(2 * M_PI) + log( SIGMA_ROBUST ) + .5 * Z_SCORE * Z_SCORE;

}

Vector Recurrent::derivative_NLL(const Vector& y_hat, const Vector& y_true) {

    const double SIGMA_SIGN = (y_hat(1) > 0);
    const double SIGMA_ROBUST = std::abs(y_hat(1)) + 1e-9; // we add a negligible value to sigma for robustness
    const double Z_SCORE = (y_true(0) - y_hat(0)) / SIGMA_ROBUST;

    // compute gradient of Gaussian Negative LogLikelihood
    Vector gradient(2, 0.);
    gradient(0) = -  Z_SCORE / SIGMA_ROBUST;
    gradient(1) = - (Z_SCORE * Z_SCORE - 1) / SIGMA_ROBUST * SIGMA_SIGN;

    return gradient;

}


// ===================================
// Optimizer
// ===================================

void Recurrent::Adam(const Vector &grad_theta_new, const Vector &grad_phi_new){

    // update the rolling mean of the gradients
    g_theta_mean = ( g_theta_mean * BETA1 ) + ( grad_theta_new * (1-BETA1) );
    g_phi_mean   = ( g_phi_mean   * BETA1 ) + ( grad_phi_new   * (1-BETA1) );

    // update the rolling variance of the gradients
    g_theta_variance = ( g_theta_variance * BETA2 ) + ( pow2(grad_theta_new) * (1-BETA2) );
    g_phi_variance   = ( g_phi_variance   * BETA2 ) + ( pow2(grad_phi_new)   * (1-BETA2) );

    // compute the corrections to the current weights according to Adam
    Vector theta_correction = g_theta_mean / (sqrt(g_theta_variance) + EPSILON_HAT);
    Vector phi_correction   = g_phi_mean   / (sqrt(g_phi_variance)   + EPSILON_HAT);

    // update the weights of the kernel U
    for (unsigned i = 0; i < U.SIZE; ++i)
        U.loc(i) -= learning_rate * theta_correction(i);

    // update the weights of the bias b
    for (unsigned i=0; i < b.SIZE; ++i)
        b(i) -= learning_rate * theta_correction(U.SIZE + i);

    // update the weights of the AR_kernels W
    for (unsigned lag = 0; lag < N_LAGS; ++lag)
        for (unsigned i = 0; i < W[lag].SIZE; ++i)
            W[lag].loc(i) -= learning_rate * theta_correction(U.SIZE + b.SIZE + lag * W[lag].SIZE + i);

    // update the weights of the kernel V
    for (unsigned i = 0; i < V.SIZE; ++i)
        V.loc(i) -= learning_rate * phi_correction(i);

    // update the weights of the bias c
    for (unsigned i=0; i < c.SIZE; ++i)
        c(i) -= learning_rate * phi_correction(V.SIZE + i);

}


// ===================================
// Fit and Predict Methods
// ===================================


std::vector<std::vector<Vector>> Recurrent::data_windowing(const std::vector<std::vector<double>>& data_x) const {

    // Store relevant quantities
    const unsigned N_ROWS = data_x.size(), N_FEATURES = data_x[0].size();

    // Convert data_x into std::vector<Vector>
    std::vector<Vector> x_mod;
    for (unsigned i = 0; i < N_ROWS; ++i) {
        Vector tmp(N_FEATURES, 0.);  // Temporary Vector of correct size
        for (unsigned j = 0; j < N_FEATURES; ++j)
            tmp(j) = data_x[i][j];  // Copy element-wise
        x_mod.push_back(tmp);
    }

    // Generate windows for x
    std::vector<std::vector<Vector>> x_windows;
    for (unsigned i = 0; i + WINDOW_SIZE <= N_ROWS; ++i)
        x_windows.emplace_back(x_mod.begin() + i, x_mod.begin() + i + WINDOW_SIZE);

    return x_windows;
}


std::pair<std::vector<std::vector<Vector>>, std::vector<Vector>>
    Recurrent::data_windowing(const std::vector<std::vector<double>>& data_x,
                              const std::vector<std::vector<double>>& data_y) const {

    // safety check
    if (data_x.size() != data_y.size()) {
        throw std::invalid_argument("data_windowing: df_x and df_y must have the same number of rows.");
    }

    // get x_windows using the simplified function
    std::vector<std::vector<Vector>> x_windows = data_windowing(data_x);

    // convert data_y into std::vector<Vector>
    const unsigned N_ROWS = data_y.size(), N_TARGET = data_y[0].size();
    std::vector<Vector> y_mod;
    for (unsigned i = 0; i < N_ROWS; ++i) {
        Vector tmp(N_TARGET, 0.);
        for (unsigned j = 0; j < N_TARGET; ++j)
            tmp(j) = data_y[i][j];
        y_mod.push_back(tmp);
    }

    // generate windows for y
    std::vector<Vector> y_windows;
    for (unsigned i = 0; i + WINDOW_SIZE <= N_ROWS; ++i) {
        y_windows.push_back(y_mod[i + WINDOW_SIZE - 1]);
    }

    return {x_windows, y_windows};
}



void Recurrent::fit(const DataFrame& df_train_x, const DataFrame& df_train_y,
                    const DataFrame& df_valid_x, const DataFrame& df_valid_y,
                    const double LEARNING_RATE, const unsigned EPOCHS, const unsigned BATCH_SIZE,
                    const unsigned PATIENCE,
                    const std::string& filename){

    // instantiate the collectors for the subsequences
    std::vector<std::vector<Vector>> x_train, x_valid;
    std::vector<Vector> y_train, y_valid;

    // create the subsequences
    std::tie(x_train, y_train) = data_windowing(df_train_x.data, df_train_y.data);
    std::tie(x_valid, y_valid) = data_windowing(df_valid_x.data, df_valid_y.data);

    // tic: start measuring training time
    auto t1 = std::chrono::high_resolution_clock::now();

    // set internal member "learning_rate" equal to the provided value
    learning_rate = LEARNING_RATE;

    // we define number of input sequences
    const unsigned N_TRAIN = x_train.size(), N_VALID = x_valid.size();

    // we create a vector with all available indices of training set (numbers from 0 to N_TRAIN-1)
    std::vector<unsigned> indices_train(N_TRAIN);
    std::iota(indices_train.begin(), indices_train.end(), 0);

    // we define the vectors that will contain the gradients: (i) of the sequence; (ii) of the minibatch
    Vector g_theta_seq(N_THETA, 0.), g_phi_seq(N_PHI, 0.);
    Vector g_theta_minibatch_avg(N_THETA, 0.), g_phi_minibatch_avg(N_PHI, 0.);

    // define metric for Early Stopping
    unsigned epochs_since_new_min = 0;
    double current_min = std::numeric_limits<double>::infinity();

    // open_out_stream
    std::ofstream outstream(filename);
    outstream << std::setprecision(utils::OUTPUT_DIGITS) << std::fixed;

    // training loop
    unsigned epoch_counter = 1; // kept outside because it is printed at the end of the function
    for (; epoch_counter <= EPOCHS && epochs_since_new_min < PATIENCE; ++epoch_counter) {

        // shuffle indices of the TRAINING SET
        utils::shuffle(indices_train, utils::rng);
        //std::shuffle(std::begin(indices_train), std::end(indices_train), utils::rng); //different in unix/osx

        // set to zero the collectors for the minibatch average
        g_theta_minibatch_avg.to_zero(); g_phi_minibatch_avg.to_zero();

        // number of elements in a minibatch
        unsigned minibatch_size = 0;

        // we loop over all available sequences
        for (unsigned seq = 0; seq < N_TRAIN; ++seq) {

            // set to zero the gradients g_theta and g_phi
            g_theta_seq.to_zero(); g_phi_seq.to_zero();

            // compute the gradients g_theta and g_phi for the processed sequence
            AAD(x_train[indices_train[seq]], y_train[indices_train[seq]], g_theta_seq, g_phi_seq);

            // sum gradients in the minibatch
            g_theta_minibatch_avg += g_theta_seq;
            g_phi_minibatch_avg   += g_phi_seq;
            ++minibatch_size;

            // if required, modify the weights as prescribed by Adam
            if ( ((seq+1) % BATCH_SIZE == 0) || ((seq+1) == N_TRAIN) ){

                // compute average on the minibatch
                g_theta_minibatch_avg /= minibatch_size;
                g_phi_minibatch_avg   /= minibatch_size;
                minibatch_size = 0;

                // run Adam optimizer
                Adam(g_theta_minibatch_avg, g_phi_minibatch_avg);

                // set to zero the minibatch average
                g_theta_minibatch_avg.to_zero(); g_phi_minibatch_avg.to_zero();

            }
        }

        // initialize loss for training set and validation set
        double loss_train = 0., loss_valid = 0;

        // depending on the number of outputs (i.e. on the point/probabilistic forecasting)
        switch(OUTPUT_SIZE){

            case 1: {

                for (unsigned i = 0; i < N_TRAIN; ++i) {
                    Vector y_train_hat = predict_aux(x_train[i]);
                    loss_train += MSE(y_train_hat, y_train[i]);
                }

                for (unsigned i = 0; i < N_VALID; ++i) {
                    Vector y_valid_hat = predict_aux(x_valid[i]);
                    loss_valid += MSE(y_valid_hat, y_valid[i]);
                }

                break;
            }

            case 2: {

                for (unsigned i = 0; i < N_TRAIN; ++i) {
                    Vector y_train_hat = predict_aux(x_train[i]);
                    loss_train += NLL(y_train_hat, y_train[i]);
                }

                for (unsigned i = 0; i < N_VALID; ++i) {
                    Vector y_valid_hat = predict_aux(x_valid[i]);
                    loss_valid += NLL(y_valid_hat, y_valid[i]);
                }

                break;
            }

            default:
                std::cerr << "Error: OUTPUT_SIZE_ can only be 1 or 2" << std::endl;
                exit(EXIT_FAILURE);
        }

        // divide by size
        loss_train /= (double) N_TRAIN;
        loss_valid /= (double) N_VALID;

        outstream << epoch_counter << "\t" << loss_train << "\t" << loss_valid << std::endl;

        // Early stopping criterion (on validation set)
        if (current_min > loss_valid) {
            current_min = loss_valid;
            epochs_since_new_min = 0;
            save_best_weights();
        }
        else {
            ++epochs_since_new_min;
        }
    }

    // restore the best weights found during the training procedure
    restore_best_weights();

    // toc: print the elapsed time for the NN training
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

    outstream << "Elapsed time: " << duration << " seconds" << std::endl;

}


DataFrame Recurrent::predict(const DataFrame& df_x) const {

    // create the subsequences
    std::vector<std::vector<Vector>> x_input = data_windowing(df_x.data);

    // create the index for the output, skipping the first (WINDOW_SIZE - 1) elements
    std::vector<std::string> time_index(df_x.index.begin() + WINDOW_SIZE - 1, df_x.index.end());

    // determine column names based on the value of OUTPUT_SIZE
    std::vector<std::string> column_names = (OUTPUT_SIZE == 1)
                                            ? std::vector<std::string>{"Mu"}
                                            : std::vector<std::string>{"Mu", "Sigma"};

    // instantiate empty DataFrame with index and column names
    DataFrame predictions(time_index, column_names);

    // fill the DataFrame with predicted values
    for (unsigned i = 0; i < x_input.size(); ++i) {
        // obtain the prediction for the i-th subsequence
        Vector pred = predict_aux(x_input[i]);
        // Fill in the corresponding entry in the
        for (unsigned j = 0; j < pred.SIZE; ++j)
            predictions.at(time_index[i], column_names[j]) = pred(j);
    }

    return predictions;
}



Vector Recurrent::predict_aux(const std::vector<Vector>& subsequence_x) const {

    // 1) we prepare vectors where to store the pre-activation state, the hidden state and the output state
    Vector a(HIDDEN_SIZE, 0.), h(HIDDEN_SIZE, 0.), y(OUTPUT_SIZE, 0.);

    // 2) we prepare a std::vector<Vectors> where to store the history of the outputs over a single sequence
    std::vector<Vector> y_history(WINDOW_SIZE, y);

    // 3) we loop over the elements in the sequence
    for (unsigned t = 0; t < WINDOW_SIZE; ++t) {

        // Compute Pre-activation state (i.e. the contribution of the First layer -- only linear part)
        a = U * subsequence_x[t] + b;
        for (unsigned i = 0; i < N_LAGS; ++i)
            if (t >= LAGS[i]) // if index is > 0
                 a += W[i] * y_history[t - LAGS[i]];

        // Compute Hidden state (i.e. after applying the nonlinear (sigmoid) Activation Function)
        h = sigmoid(a);

        // Compute Output state (i.e. the Second layer -- linear part)
        y = V * h + c;

        //update stored outputs
        y_history[t] = y;

    }

    // return the final prediction
    return y_history[WINDOW_SIZE-1];

}


// ===================================
// Early Stopping
// ===================================

void Recurrent::save_best_weights(){
    U_best = U;
    V_best = V;
    b_best = b;
    c_best = c;
    for (unsigned k = 0; k < N_LAGS; ++k) {
        W_best[k] = W[k];
    }
}

void Recurrent::restore_best_weights(){
    U = U_best;
    V = V_best;
    b = b_best;
    c = c_best;
    for (unsigned k = 0; k < N_LAGS; ++k) {
        W[k] = W_best[k];
    }
}



// ===================================
// Gradient Algorithms: RTRL
// ===================================

void Recurrent::RTRL(const std::vector<Vector>& x_input, const Vector& y_true,
                     Vector& gradient_theta, Vector& gradient_phi){

    // store the length of each sequence (i.e. TAU)
    const unsigned TAU = x_input.size();

    // Memory allocation:

    // 1) we prepare a vector where to store the pre-activation state and one where to store the hidden state
    Vector a(HIDDEN_SIZE, 0.), h(HIDDEN_SIZE, 0.), y(OUTPUT_SIZE, 0.);

    // 2) we prepare a vector where to store the Jacobian matrix dy/dtheta and dy/dphi at time t
    Matrix jacobian_theta(OUTPUT_SIZE, N_THETA, 0.), jacobian_phi(OUTPUT_SIZE, N_PHI, 0.);

    // 3) memory management could be more optimised (e.g. CircularBuffer), but we opt for a more readable code.
    //    Indeed, we do not need the entire history TAU, but just the last N_LAGS elements.
    std::vector<Matrix> jacobian_theta_history(TAU, jacobian_theta), jacobian_phi_history(TAU, jacobian_phi  );
    std::vector<Vector> y_history(TAU, y);

    // for all the elements in the sequence
    for (unsigned t = 0; t < TAU; ++t) {

        /** COMPUTE FORECAST **/

        // Compute Pre-activation state (i.e. the contribution of the First layer -- only affine part)
        a = U * x_input[t] + b;
        for (unsigned i = 0; i < N_LAGS; ++i)
            if( t >= LAGS[i]) // if index is >= 0
                a += W[i] * y_history[ t - LAGS[i] ];

        // Compute Hidden state (i.e. after applying the Sigmoid Activation Function)
        h = sigmoid(a);

        // Compute Output state (i.e. the Second layer -- which is only affine)
        y = V * h + c;

        /** COMPUTE GRADIENTS (Jacobian Matrices) **/

        // we compute the matrix product V * A_t
        Vector A_t = derivative_sigmoid(h);
        Matrix VA = mult_diagonal(V, A_t);

        // set to zero the matrices where we store jacobian matrices at time t
        jacobian_theta.to_zero(); jacobian_phi.to_zero();

        // compute contribution to the partial derivatives of the element being processed (i.e. that at time t)
        partial_derivatives_RTRL(jacobian_theta, jacobian_phi, x_input[t], h, y_history, VA, t);

        for (unsigned i = 0; i < N_LAGS; ++i){

            if (t < LAGS[i])
                continue;

            // store for convenience
            Matrix VAW = VA * W[i];

            // add the contributions of the previous total derivatives (the two sums)
            jacobian_theta += VAW * jacobian_theta_history[ t - LAGS[i] ];
            jacobian_phi   += VAW * jacobian_phi_history  [ t - LAGS[i] ];

        }

        //update stored values and Jacobian matrices
        jacobian_theta_history[t] = jacobian_theta;
        jacobian_phi_history[t]   = jacobian_phi;
        y_history[t] = y;

    }

    // we compute the gradient of the loss with respect to the final output
    Vector dL(OUTPUT_SIZE, 0.);
    switch(OUTPUT_SIZE){
        case 1:
            dL = derivative_MSE(y_history[TAU-1], y_true); break;
        case 2:
            dL = derivative_NLL(y_history[TAU-1], y_true); break;
        default:
            std::cerr << "Error: OUTPUT_SIZE_ can only be 1 or 2" << std::endl;
            exit(EXIT_FAILURE);
    }

    // prepare a vector where to store the Jacobian matrix dy/dtheta and dy/dphi at time t
    gradient_theta = dL * jacobian_theta;
    gradient_phi = dL * jacobian_phi;

    //gradient_theta.print();
    //gradient_phi.print();

}


// Partial Derivative: a time-efficient implementation of the computation of partial derivatives
void Recurrent::partial_derivatives_RTRL(Matrix& jacobian_theta, Matrix& jacobian_phi,
                                         const Vector& x, const Vector& h, const std::vector<Vector>& y_history,
                                         const Matrix& VA, unsigned t){

    /** FIRST LAYER **/

    // kernel U
    for (unsigned i = 0; i < OUTPUT_SIZE; ++i)
        for (unsigned j = 0;  j < HIDDEN_SIZE; ++j)
            for (unsigned k = 0; k < INPUT_SIZE; ++k)
                jacobian_theta(i, j * INPUT_SIZE + k) = VA(i, j) * x(k);

    // bias b
    for (unsigned i = 0; i < OUTPUT_SIZE; ++i)
        for (unsigned j = 0;  j < HIDDEN_SIZE; ++j)
            jacobian_theta(i, U.SIZE + j) = VA(i,j);

    // AutoRegressive kernels W_l
    for (unsigned lag = 0; lag < N_LAGS; ++lag) {
        const unsigned OFFSET = U.SIZE + b.SIZE + lag * W[lag].SIZE;
        for (unsigned i = 0; i < OUTPUT_SIZE; ++i)
            for (unsigned j = 0; j < HIDDEN_SIZE; ++j)
                for (unsigned k = 0; k < OUTPUT_SIZE; ++k)
                    if (t >= LAGS[lag])
                        jacobian_theta(i, OFFSET + j * OUTPUT_SIZE + k) = VA(i, j) * y_history[t - LAGS[lag]](k);
    }


    /** SECOND LAYER **/

    // kernel V
    for (unsigned i = 0; i < OUTPUT_SIZE; ++i)
        for (unsigned j = 0; j < HIDDEN_SIZE; ++j)
            jacobian_phi(i, i * HIDDEN_SIZE + j) = h(j);

    // bias c
    for (unsigned i = 0; i < c.SIZE; ++i)
        jacobian_phi(i, V.SIZE + i) = 1;

}


// ===================================
// Gradient Algorithms: AAD
// ===================================

void Recurrent::AAD(const std::vector<Vector>& x_input, const Vector& y_true,
                    Vector& gradient_theta, Vector& gradient_phi) {

    // store the length of each sequence (i.e. TAU)
    const unsigned TAU = x_input.size();

    // Memory allocation:

    // 1) we prepare a vector where to store the pre-activation state and one where to store the hidden state
    Vector a(HIDDEN_SIZE, 0.), h(HIDDEN_SIZE, 0.), y(OUTPUT_SIZE, 0.);

    // 2) we prepare a vector where to store the Jacobian matrix dy/dtheta and dy/dphi at time t
    Matrix jacobian_theta(OUTPUT_SIZE, N_THETA, 0.), jacobian_phi(OUTPUT_SIZE, N_PHI, 0.);

    // 3) we prepare a std::vector of vectors where to store the history of h and y.
    //    In this case, it is mandatory to store the entire history over the whole sequence TAU (vs RTRL case).
    std::vector<Vector> h_history(TAU, h), y_history(TAU, y);

    /** FORWARD PASS **/
    // for all the elements in the sequence
    for (unsigned t = 0; t < TAU; ++t) {

        // Compute Pre-activation state (i.e. the contribution of the First layer -- only affine part)
        a = U * x_input[t] + b;
        for (unsigned i = 0; i < N_LAGS; ++i)
            if( t >= LAGS[i]) // if index is > 0
                a += W[i] * y_history[ t - LAGS[i] ];

        // Compute Hidden state (i.e. after applying the Sigmoid Activation Function)
        h = sigmoid(a);

        // Compute Output state (i.e. the Second layer -- which is only affine)
        y = V * h + c;

        //update stored values and Jacobian matrices
        h_history[t] = h;
        y_history[t] = y;

    }

    // we compute the gradient of the loss with respect to the final output
    Vector dL(OUTPUT_SIZE, 0.);
    //double L;
    //L = MSE(y_history[TAU-1], y_true);

    switch(OUTPUT_SIZE){
        case 1:
            dL = derivative_MSE(y_history[TAU-1], y_true); break;
        case 2:
            dL = derivative_NLL(y_history[TAU-1], y_true); break;
        default:
            std::cerr << "Error: OUTPUT_SIZE_ can only be 1 or 2" << std::endl;
            exit(EXIT_FAILURE);
    }


    /** BACKWARD PASS **/

    // we instantiate a vector containing all the gradients of the output wrt the loss
    std::vector<Vector> g_history(TAU, Vector(OUTPUT_SIZE, 0.));

    // we set the last element in g_history equal to the derivative of the loss function w.r.t. y_TAU
    g_history[TAU-1] = dL;

    for (unsigned s = 0; s < TAU; ++s) {
        unsigned t = TAU - 1 - s;

        // we compute the matrix product z_t = [V * A_t]^T * g_t
        Vector A_t = derivative_sigmoid(h_history[t]);
        Matrix VA  = mult_diagonal(V, A_t);
        Vector z_t = g_history[t] * VA;

        // update g_history
        for (unsigned i = 0; i < N_LAGS; ++i) {
            if (t >= LAGS[i])
                g_history[t - LAGS[i]] += z_t * W[i];
        }

        partial_derivatives_AAD(gradient_theta, gradient_phi, x_input[t], h_history[t], y_history, z_t, g_history[t], t);

    }

}


// Partial Derivative: a time-efficient implementation of the computation of partial derivatives
void Recurrent::partial_derivatives_AAD( Vector& gradient_theta, Vector& gradient_phi,
                                         const Vector& x, const Vector& h, const std::vector<Vector>& y_history,
                                         const Vector& z, const Vector& g, unsigned t){

    /** FIRST LAYER **/

    // kernel
    for (unsigned j = 0; j < HIDDEN_SIZE; ++j)
        for (unsigned k = 0; k < INPUT_SIZE; ++k)
            gradient_theta(j*INPUT_SIZE + k) += z(j) * x(k);

    // bias
    for (unsigned j = 0; j < HIDDEN_SIZE; ++j)
        gradient_theta(U.SIZE + j) += z(j);

    // AR kernels
    for (unsigned lag = 0; lag < N_LAGS; ++lag) {
        const unsigned OFFSET = U.SIZE + b.SIZE + lag * W[lag].SIZE;
        for (unsigned j = 0; j < HIDDEN_SIZE; ++j)
            for (unsigned k = 0; k < OUTPUT_SIZE; ++k)
                if (t >= LAGS[lag])
                    gradient_theta(OFFSET + j * OUTPUT_SIZE + k) += z(j) * y_history[t - LAGS[lag]](k);
    }

    /** SECOND LAYER **/

    // kernel
    for (unsigned j = 0; j < OUTPUT_SIZE; ++j)
        for (unsigned k = 0; k < HIDDEN_SIZE; ++k)
            gradient_phi(j * HIDDEN_SIZE + k) += g(j) * h(k);

    // bias
    for (unsigned j = 0; j < OUTPUT_SIZE; ++j)
        gradient_phi(V.SIZE + j) += g(j);

}

