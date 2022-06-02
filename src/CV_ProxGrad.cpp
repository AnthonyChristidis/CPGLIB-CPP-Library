/*
* ===========================================================
* File Type: CPP
* File Name: CV_ProxGrad.cpp
* Package Name: CPGLIB
*
* Created by Anthony-A. Christidis.
* Copyright © Anthony-A. Christidis. All rights reserved.
* ===========================================================
*/

#include <RcppArmadillo.h>

#include "config.h"

#include "ProxGrad.hpp"
#include "CV_ProxGrad.hpp"

#include <vector>

// Constructor - with data
CV_ProxGrad::CV_ProxGrad(arma::mat & x, arma::vec & y,
                         arma::uword & type,
                         arma::uword & include_intercept,
                         double & alpha_s, 
                         arma::uword & n_lambda_sparsity,
                         double & tolerance, arma::uword & max_iter,
                         arma::uword & n_folds,
                         arma::uword & n_threads): 
  x(x), y(y),
  type(type),
  include_intercept(include_intercept),
  alpha_s(alpha_s), 
  n_lambda_sparsity(n_lambda_sparsity), 
  tolerance(tolerance), max_iter(max_iter),
  n_folds(n_folds),
  n_threads(n_threads){
  
  // Initialization of the weighted elastic net models (one for each fold)
  Initialize();
}


// Function to initial the object characteristics
void CV_ProxGrad::Initialize(){
  
  // Setting the parameters for the data dimension
  n = x.n_rows;
  p = x.n_cols;
  
  // Initializing the size of the parameter variables for CV object
  intercepts = arma::zeros(n_lambda_sparsity);
  betas = arma::zeros(p, n_lambda_sparsity);
  cv_errors_sparsity = arma::zeros(n_lambda_sparsity);
  cv_errors_sparsity_mat = arma::zeros(n_lambda_sparsity, n_folds);

  // Computing the grid for lambda_sparsity
  if(n > p)
    eps = 1e-4;
  else
    eps = 1e-2;
  Compute_Lambda_Sparsity_Grid();
  
  // Setting function pointers for the deviance
  if(type==1){ // Linear GLM
    Compute_Deviance = &CV_ProxGrad::Linear_Deviance; 
  }
  else if(type==2){ // Logistic GLM
    Compute_Deviance = &CV_ProxGrad::Logistic_Deviance;
  }
}

// Method to get the grid for lambda_sparsity
void CV_ProxGrad::Compute_Lambda_Sparsity_Grid(){
  
  // Standardization of design matrix
  arma::rowvec mu_x = arma::mean(x);
  arma::rowvec sd_x = arma::stddev(x, 1);
  arma::mat x_std = x;
  x_std.each_row() -= mu_x;
  x_std.each_row() /= sd_x;
  
  // Maximum lambda_sparsity that kills all variables
  double lambda_sparsity_max;
  lambda_sparsity_max = (1/alpha_s)*arma::max(abs(y.t()*x_std))/n;
  lambda_sparsity_grid =  arma::exp(arma::linspace(std::log(eps*lambda_sparsity_max), std::log(lambda_sparsity_max), n_lambda_sparsity));
}

// Function to create the folds
arma::uvec CV_ProxGrad::Set_Diff(const arma::uvec & big, const arma::uvec & small){
  
  // Find set difference between a big and a small set of variables.
  // Note: small is a subset of big (both are sorted).
  arma::uword m = small.n_elem;
  arma::uword n = big.n_elem;
  arma::uvec test = arma::uvec(n, arma::fill::zeros);
  arma::uvec zeros =arma:: uvec(n - m, arma::fill::zeros);
  
  for (arma::uword j = 0 ; j < m ; j++){
    test[small[j]] = small[j];
  }
  
  test = big - test;
  if(small[0] != 0){
    test[0] = 1;
  }
  zeros = find(test != 0);
  return(zeros);
}

// Private function to compute the CV-MSPE over the folds
void CV_ProxGrad::Compute_CV_Deviance_Sparsity(arma::uword & sparsity_ind, arma::uword & fold_ind,
                                               arma::mat x_test, arma::vec y_test,
                                               double intercept, arma::vec betas){
  
  // Computing the CV-Error over the folds
  for(arma::uword fold_ind=0; fold_ind<n_folds; fold_ind++)
    cv_errors_sparsity_mat(sparsity_ind, fold_ind) = (*Compute_Deviance)(x_test, y_test, intercept, betas);
  
}

// Functions to set new data
void CV_ProxGrad::Set_X(arma::mat & x){
  
  this->x = x;
  // Standardization of design matrix
  mu_x = arma::mean(x);
  sd_x = arma::stddev(x, 1);
  x.each_row() -= mu_x;
  x.each_row() /= sd_x;
  
  // Augmented matrix for optimization
  if(include_intercept)
    x_std_aug = arma::join_rows(x, arma::zeros(n, 1));
  else
    x_std_aug = arma::join_rows(x, arma::zeros(n, 1));
}
void CV_ProxGrad::Set_Y(arma::vec & y){
  this->y = y;
}

// Functions to set maximum number of iterations and tolerance
void CV_ProxGrad::Set_Max_Iter(arma::uword & max_iter){
  this->max_iter = max_iter;
}
void CV_ProxGrad::Set_Tolerance(double & tolerance){
  this->tolerance = tolerance;
}

// Functions to set and get alpha_s
void CV_ProxGrad::Set_Alpha_Sparsity(double alpha_s){
  this->alpha_s = alpha_s;
}
double CV_ProxGrad::Get_Alpha_Sparsity(){
  return(this->alpha_s);
}

// Method to get the grid of lambda_sparsity
arma::vec CV_ProxGrad::Get_Lambda_Sparsity_Grid(){
  return(lambda_sparsity_grid);
}

// Cross-validation - Sparsity
arma::vec CV_ProxGrad::Get_CV_Error_Sparsity(){
  return(cv_errors_sparsity);
}

// Optimal penalty parameter - Sparsity
double CV_ProxGrad::Get_lambda_sparsity_opt(){
  return(lambda_sparsity_opt);
}

// Methods to return coefficients
arma::mat CV_ProxGrad::Get_Coef(){
  return(betas);
}
arma::vec CV_ProxGrad::Get_Intercept(){
  return(intercepts);
}

// Optimal sparsity parameter
arma::uword CV_ProxGrad::Get_Optimal_Index_Sparsity(){
  return(cv_errors_sparsity.index_min());
}

// Computing the solutions over a grid for folds. Grid is either for the sparsity or the diverity (one of them is fixed)
void CV_ProxGrad::Compute_CV_Grid(arma::uvec & sample_ind, arma::uvec & fold_ind){
  
  // Looping over the folds
  # pragma omp parallel for num_threads(n_threads)
  for(arma::uword fold=0; fold<n_folds; fold++){ 
    
    // Get test and training samples
    arma::uvec test = arma::linspace<arma::uvec>(fold_ind[fold],
                                                 fold_ind[fold + 1] - 1,
                                                 fold_ind[fold + 1] - fold_ind[fold]);
    arma::uvec train = Set_Diff(sample_ind, test);

    // Initialization of the WEN objects (with the maximum value of lambda_sparsity_grid)
    ProxGrad ProxGrad_fold = ProxGrad(x.rows(train), y.elem(train),
                                      type, 
                                      include_intercept,
                                      alpha_s, 
                                      lambda_sparsity_grid[n_lambda_sparsity-1],
                                      tolerance, max_iter);

    // Looping over the different sparsity penalty parameters
    for(arma::uword sparsity_ind=0; sparsity_ind<=n_lambda_sparsity-1; sparsity_ind++){
      
      // Setting the lambda_sparsity value
      ProxGrad_fold.Set_Lambda_Sparsity(lambda_sparsity_grid[sparsity_ind]);
      // Computing the betas for the fold (new lambda_sparsity)
      ProxGrad_fold.Compute_Coef();
      
      // Computing the deviance for the fold (new lambda_sparsity)
      Compute_CV_Deviance_Sparsity(sparsity_ind, fold,
                                   x.rows(test), y.elem(test),
                                   ProxGrad_fold.Get_Intercept_Scaled(), ProxGrad_fold.Get_Coef_Scaled());
      
    } // End of loop over the sparsity parameter values
    
  } // End of loop over the folds
  
  // Storing the optimal sparsity parameters
  cv_errors_sparsity = arma::mean(cv_errors_sparsity_mat, 1);
  index_sparsity_opt = cv_errors_sparsity.index_min();
  lambda_sparsity_opt = lambda_sparsity_grid[index_sparsity_opt];
  cv_opt_new = arma::min(cv_errors_sparsity);
}

// Coordinate descent algorithms for coefficients
void CV_ProxGrad::Compute_CV_Betas(){
  
  // Creating indices for the folds of the data
  arma::uvec sample_ind = arma::linspace<arma::uvec>(0, n-1, n);
  arma::uvec fold_ind = arma::linspace<arma::uvec>(0, n, n_folds+1);

  // Initial cycle
  Compute_CV_Grid(sample_ind, fold_ind);

  // Computing the parameters for the full data
  ProxGrad ProxGrad_Full = ProxGrad(x, y,
                                    type, 
                                    include_intercept,
                                    alpha_s,
                                    lambda_sparsity_grid[0],
                                    tolerance, max_iter);
  
  // Looping over the different sparsity penalty parameters
  for(arma::uword sparsity_ind=0; sparsity_ind<=n_lambda_sparsity-1; sparsity_ind++){
    
    // Setting the lambda_sparsity value
    ProxGrad_Full.Set_Lambda_Sparsity(lambda_sparsity_grid[sparsity_ind]);
    // Computing the betas for the fold (new lambda_sparsity)
    ProxGrad_Full.Compute_Coef();
    // Storing the full data models
    intercepts[sparsity_ind] =  ProxGrad_Full.Get_Intercept_Scaled();
    betas.col(sparsity_ind) = ProxGrad_Full.Get_Coef_Scaled();
    
  } // End of loop over the sparsity parameter values
}

CV_ProxGrad::~CV_ProxGrad(){
  // Class destructor
}


/*
* -----------------------------
* Static Functions - Deviance
* -----------------------------
*/

// Linear Deviance (MSPE)
double CV_ProxGrad::Linear_Deviance(arma::mat & x, arma::vec & y,
                                    double & intercept, arma::vec & betas){
  
  arma::vec linear_fit = intercept + x*betas;
  return(arma::accu(arma::square(linear_fit - y))/2);
}

// Logistic Deviance
double CV_ProxGrad::Logistic_Deviance(arma::mat & x, arma::vec & y,
                                      double & intercept, arma::vec & betas){
  
  arma::vec linear_fit = intercept + x*betas;
  return(2*arma::accu(arma::log(1 + arma::exp(linear_fit)) - linear_fit % y));
}







