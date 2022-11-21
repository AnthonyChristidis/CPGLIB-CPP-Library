/*
* ===========================================================
* File Type: CPP
* File Name: CV_ProxGrad_Main.cpp
* Package Name: CPGLIB
*
* Created by Anthony-A. Christidis.
* Copyright (c) Anthony-A. Christidis. All rights reserved.
* ===========================================================
*/

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include "config.h"

#include "CV_ProxGrad.hpp" 

// [[Rcpp::export]]
Rcpp::List CV_ProxGrad_Main(arma::mat & x, arma::vec & y,  
                            arma::uword & type, 
                            arma::uword & include_intercept, 
                            double & alpha_s,
                            arma::uword & n_lambda_sparsity,
                            double & tolerance, arma::uword & max_iter,
                            arma::uword & n_folds,
                            arma::uword & n_threads){
  
  CV_ProxGrad model = CV_ProxGrad(x, y, 
                                  type, include_intercept, 
                                  alpha_s, 
                                  n_lambda_sparsity, 
                                  tolerance, max_iter,
                                  n_folds,
                                  n_threads);
  
  // Computing coefficients
  model.Compute_CV_Betas();
  
  // Output formatting
  Rcpp::List output;
  output["Lambda_Sparsity"] = model.Get_Lambda_Sparsity_Grid();
  output["Lambda_Sparsity_Min"] = model.Get_lambda_sparsity_opt();
  output["CV_Errors"] = model.Get_CV_Error_Sparsity();
  output["Optimal_Index"] = (model.Get_CV_Error_Sparsity()).index_min() + 1;
  output["Intercept"] = model.Get_Intercept();
  output["Betas"] = model.Get_Coef();
  return(output);
}