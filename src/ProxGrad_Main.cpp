/*
* ===========================================================
* File Type: CPP
* File Name: ProxGrad_Main.cpp
* Package Name: CPGLIB
*
* Created by Anthony-A. Christidis.
* Copyright © Anthony-A. Christidis. All rights reserved.
* ===========================================================
*/

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include "config.h"

#include "ProxGrad.hpp" 

// [[Rcpp::export]]
Rcpp::List ProxGrad_Main(arma::mat & x, arma::vec & y,  
                         arma::uword & type, 
                         arma::uword & include_intercept, 
                         double & alpha_s,
                         double & lambda_sparsity,
                         double & tolerance, arma::uword & max_iter){
  
  ProxGrad model = ProxGrad(x, y, 
                            type, include_intercept, 
                            alpha_s, 
                            lambda_sparsity, 
                            tolerance, max_iter);
  
  // Computing coefficients
  model.Compute_Coef();
  
  // Output formatting
  Rcpp::List output;
  output["Intercept"] = model.Get_Intercept_Scaled();
  output["Betas"] = model.Get_Coef_Scaled();
  output["Objective"] = model.Get_Objective_Value();
  return(output);
}