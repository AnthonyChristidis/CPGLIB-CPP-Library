/*
* ===========================================================
* File Type: CPP
* File Name: CV_CPGLIB_Main.cpp
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
#include "CV_CPGLIB.hpp" 

// [[Rcpp::export]]
Rcpp::List CV_CPGLIB_Main(arma::mat & x, arma::vec & y,  
                          arma::uword & type, 
                          arma::uword & G,
                          arma::uword & full_diversity,
                          arma::uword & include_intercept, 
                          double & alpha_s, double & alpha_d,
                          arma::uword & n_lambda_sparsity, arma::uword & n_lambda_diversity,
                          double & tolerance, arma::uword & max_iter,
                          arma::uword & n_folds,
                          arma::uword & n_threads){
  
  
  
  // Case for a single model
  if(G==1){ 
    
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
  else{ // Case for more than one model
    
    CV_CPGLIB model = CV_CPGLIB(x, y, 
                                type,
                                G,
                                include_intercept, 
                                alpha_s, alpha_d,
                                n_lambda_sparsity, n_lambda_diversity,
                                tolerance, max_iter,
                                n_folds,
                                n_threads);
    
    // Computing coefficients
    if(full_diversity)
      model.Compute_CV_Betas_Full_Diversity(); else
        model.Compute_CV_Betas();
    
    // Output formatting
    Rcpp::List output;
    output["Lambda_Sparsity"] = model.Get_Lambda_Sparsity_Grid();
    output["Lambda_Sparsity_Min"] = model.Get_Lambda_Sparsity_Opt();
    output["Lambda_Diversity_Min"] = model.Get_Lambda_Diversity_Opt();
    output["CV_Errors"] = model.Get_CV_Error_Sparsity();
    output["Optimal_Index"] = (model.Get_CV_Error_Sparsity()).index_min() + 1;
    output["Intercept"] = model.Get_Intercept();
    output["Betas"] = model.Get_Coef();
    return(output);
  }
}