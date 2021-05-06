/*
* ===========================================================
* File Type: CPP
* File Name: CV_CPGLIB.cpp
* Package Name: SplitGLM
*
* Created by Anthony-A. Christidis.
* Copyright © Anthony-A. Christidis. All rights reserved.
* ===========================================================
*/

#include <RcppArmadillo.h>

#include "config.h"

#include "CV_ProxGrad.hpp"
#include "CPGLIB.hpp"
#include "CV_CPGLIB.hpp"

#include <vector>

// Constnat - Computation
static const arma::uword GRID_INTERACTION_MAX_COUNTER = 5;
static const double CV_ITERATIONS_TOLERANCE = 1e-3;
static const double CV_ITERATIONS_MAX = 10;
static const double MAX_DIVERSITY_TOLERANCE = 1e-5; 

// Constructor - with data
CV_CPGLIB::CV_CPGLIB(arma::mat & x, arma::vec & y,
                     arma::uword & type,
                     arma::uword & G, 
                     arma::uword & include_intercept,
                     double & alpha_s, double & alpha_d,
                     arma::uword & n_lambda_sparsity, arma::uword & n_lambda_diversity,
                     arma::uword & balanced_cycling,
                     arma::uword & permutate_search,
                     arma::uword & acceleration,
                     double & tolerance, arma::uword & max_iter,
                     arma::uword & n_folds,
                     arma::uword & n_threads): 
  x(x), y(y),
  type(type),
  G(G), 
  include_intercept(include_intercept), 
  alpha_s(alpha_s), alpha_d(alpha_d),
  n_lambda_sparsity(n_lambda_sparsity), n_lambda_diversity(n_lambda_diversity),
  balanced_cycling(balanced_cycling),
  permutate_search(permutate_search), acceleration(acceleration),
  tolerance(tolerance), max_iter(max_iter),
  n_folds(n_folds),
  n_threads(n_threads){

  // Initialization of the weighted elastic net models (one for each fold)
  Initialize();
}

// Function to initial the object characteristics
void CV_CPGLIB::Initialize(){
  
  // Setting the parameters for the data dimension
  n = x.n_rows;
  p = x.n_cols;
  
  // Initializing the size of the parameter variables for CV object
  intercepts = arma::zeros(G, n_lambda_sparsity);
  betas = arma::zeros(p, G, n_lambda_sparsity);
  cv_errors_sparsity = arma::zeros(n_lambda_sparsity);
  cv_errors_diversity = arma::zeros(n_lambda_diversity);
  
  // Computing the grid for lambda_sparsity
  if(n>p){
    eps_sparsity = 1e-4;
    eps_diversity = 7.5e-4;
  }
  else{
    eps_sparsity = 1e-2;
    eps_diversity = 7.5e-2;
  }
  Compute_Lambda_Sparsity_Grid();
  
  // Setting function pointers for the deviance
  if(type==1){ // Linear Model
    Compute_Deviance = &CV_CPGLIB::Linear_Deviance;
  }
  else if(type==2){ // Logistic Regression
    Compute_Deviance = &CV_CPGLIB::Logistic_Deviance;
  }
  else if(type==3){ // Gamma GLM
    Compute_Deviance = &CV_CPGLIB::Gamma_Deviance;
  }
  else if(type==4){ // Poisson GLM
    Compute_Deviance = &CV_CPGLIB::Poisson_Deviance;
  }
}

// Function to create the folds
arma::uvec CV_CPGLIB::Set_Diff(const arma::uvec & big, const arma::uvec & small){
  
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

// Function that checks if there are interactions between groups in the matrix of betas
bool CV_CPGLIB::Check_Interactions_Beta(arma::mat beta){
  
  arma::uword p = beta.n_rows;
  bool interactions = false;
  for(arma::uword beta_ind = 0; beta_ind < p; beta_ind++){
    arma::mat temp = arma::nonzeros(beta.row(beta_ind));
    if(temp.n_rows > 1){
      return(true);
    }
  }
  return(interactions);
}

// Function to returns a vector with ones corresponding to the betas that have interactions.
arma::uvec CV_CPGLIB::Check_Interactions(arma::cube & betas){
  
  arma::vec checks = arma::zeros(betas.n_slices, 1);
  arma::vec all_ones = arma::ones(betas.n_slices, 1);
  for(arma::uword beta_ind = 0; beta_ind < betas.n_slices; beta_ind++){
    checks(beta_ind) = Check_Interactions_Beta(betas.slice(beta_ind));
  }
  return(checks==all_ones);
}

// Method to get a diversity penalty parameter that kills all model interactions
double CV_CPGLIB::Get_Lambda_Diversity_Max(){
  
  // Initial guess for the diversity penalty
  double lambda_diversity_max = 3*G;
  
  // Split model to determine the maximum diversity parameter
  CPGLIB beta_grid = CPGLIB(x, y,
                            type,
                            G, 
                            include_intercept,
                            alpha_s, alpha_d, 
                            lambda_sparsity_opt, lambda_diversity_max,
                            permutate_search,
                            acceleration,
                            MAX_DIVERSITY_TOLERANCE, max_iter);
  beta_grid.Cycle_Groups_Balanced();
  arma::uword counter = 0;
  // While interactions remain, increase lambda_diversity_max by scaling it by a constant factor of two
  while(Check_Interactions_Beta(beta_grid.Get_Coef_Scaled()) & (counter<=GRID_INTERACTION_MAX_COUNTER)){
    
    counter++;
    lambda_diversity_max = lambda_diversity_max*2;
    beta_grid.Set_Lambda_Diversity(lambda_diversity_max);
    beta_grid.Cycle_Groups_Balanced();
  }
  
  // If we could not kill all the interactions
  if(Check_Interactions_Beta(beta_grid.Get_Coef_Scaled())){
    
    Rcpp::warning("Failure to find lambda_diversity that kills all interactions.");
  }
  else{
    
    // Initial grid for maximum diversity search
    lambda_diversity_grid = arma::exp(arma::linspace(std::log(eps_diversity*lambda_diversity_max), 
                                                     std::log(lambda_diversity_max), 
                                                     n_lambda_diversity));
    
    // Computing the beta coefficients for the candidates in the grid
    for(int diversity_ind=n_lambda_diversity-1; diversity_ind>=0; diversity_ind--){
      
      beta_grid.Set_Lambda_Diversity(lambda_diversity_grid[diversity_ind]);
      beta_grid.Cycle_Groups_Balanced();
      if(Check_Interactions_Beta(beta_grid.Get_Coef_Scaled()))
        break;
    }
    
    // Find smallest lambda_diversity in the grid such that there are no interactions
    lambda_diversity_max = beta_grid.Get_Lambda_Diversity();
    lambda_diversity_grid = arma::exp(arma::linspace(std::log(eps_diversity*lambda_diversity_max), 
                                                     std::log(lambda_diversity_max), 
                                                     n_lambda_diversity));
  }
  
  // Return the diversity penalty parameter 
  // std::cout << "lambda_diversity_max: " << lambda_diversity_max << std::endl;
  return(lambda_diversity_max);
}

// Method to set lambda to new value and return current lambda
void CV_CPGLIB::Compute_Lambda_Sparsity_Grid(){
  
  // Standardization of design matrix
  arma::rowvec mu_x = arma::mean(x);
  arma::rowvec sd_x = arma::stddev(x, 1);
  arma::mat x_std = x;
  x_std.each_row() -= mu_x;
  x_std.each_row() /= sd_x;
  
  // Maximum lambda_sparsity that kills all variables
  double lambda_sparsity_max;
  lambda_sparsity_max = (1/alpha_s)*arma::max(abs(y.t()*x_std))/n;
  lambda_sparsity_grid =  arma::exp(arma::linspace(std::log(eps_sparsity*lambda_sparsity_max), std::log(lambda_sparsity_max), n_lambda_sparsity));
}

// Method to set lambda to new value and return current lambda
void CV_CPGLIB::Compute_Lambda_Diversity_Grid(){ 
  
  // Initial guess for the diversity penalty
  double lambda_diversity_max = Get_Lambda_Diversity_Max();
  
  // Current grid
  lambda_diversity_grid = arma::exp(arma::linspace(std::log(eps_diversity*lambda_diversity_max), std::log(lambda_diversity_max), n_lambda_diversity));
}

// Private function to compute the CV-MSPE over the folds
void CV_CPGLIB::Compute_CV_Deviance_Sparsity(int & sparsity_ind,
                                             arma::mat x_test, arma::vec y_test,
                                             arma::vec intercept, arma::mat betas){
    
  // Computing the CV-Error over the folds
  for(arma::uword fold_ind=0; fold_ind<n_folds; fold_ind++)
    cv_errors_sparsity[sparsity_ind] += (*Compute_Deviance)(x_test, y_test, intercept, betas) / n;
  
}

// Private function to compute the CV-MSPE over the folds
void CV_CPGLIB::Compute_CV_Deviance_Diversity(int & diversity_ind,
                                              arma::mat x_test, arma::vec y_test,
                                              arma::vec intercept, arma::mat betas){
  
  // Computing the CV-Error over the folds
  for(arma::uword fold_ind=0; fold_ind<n_folds; fold_ind++)
    cv_errors_diversity[diversity_ind] += (*Compute_Deviance)(x_test, y_test, intercept, betas) / n;
  
}

// Functions to set new data
void CV_CPGLIB::Set_X(arma::mat & x){
  this->x = x;
}
void CV_CPGLIB::Set_Y(arma::vec & y){
  this->y = y;
}

// Method to set alpha_s to new value and return current alpha_s
void CV_CPGLIB::Set_Alpha_Sparsity(double alpha_s){
  this->alpha_s = alpha_s;
}
double CV_CPGLIB::Get_Alpha_Sparsity(){
  return(alpha_s);
}
// Method to set alpha_s to new value and return current alpha_s
void CV_CPGLIB::Set_Alpha_Diversity(double alpha_d){
  this->alpha_d = alpha_d;
}
double CV_CPGLIB::Get_Alpha_Diversity(){
  return(alpha_d);
}

// Method to get the grid of lambda_sparsity
arma::vec CV_CPGLIB::Get_Lambda_Sparsity_Grid(){
  return(lambda_sparsity_grid);
}
// Method to get the grid of lambda_diversity
arma::vec CV_CPGLIB::Get_Lambda_Diversity_Grid(){
  return(lambda_diversity_grid);
}

// Functions to set maximum number of iterations and tolerance
void CV_CPGLIB::Set_Max_Iter(arma::uword & max_iter){
  this->max_iter = max_iter;
}
void CV_CPGLIB::Set_Tolerance(double & tolerance){
  this->tolerance = tolerance;
}

// Cross-validation - Sparsity
arma::vec CV_CPGLIB::Get_CV_Error_Sparsity(){
  return(cv_errors_sparsity);
}
// Cross-validation - Diversity
arma::vec CV_CPGLIB::Get_CV_Error_Diversity(){
  return(cv_errors_diversity);
}

// Optimal penalty parameter - Sparsity
double CV_CPGLIB::Get_Lambda_Sparsity_Opt(){
  return(lambda_sparsity_opt);
}
// Optimal penalty parameter - Diversity
double CV_CPGLIB::Get_Lambda_Diversity_Opt(){
  return(lambda_diversity_opt);
}

// Methods to return coefficients
arma::cube CV_CPGLIB::Get_Coef(){
  return(betas);  
}
arma::mat CV_CPGLIB::Get_Intercept(){
  return(intercepts);
}

// Optimal sparsity parameter
arma::uword CV_CPGLIB::Get_Optimal_Index_Sparsity(){
  return(cv_errors_sparsity.index_min());
}
// Optimal diversity parameter
arma::uword CV_CPGLIB::Get_Optimal_Index_Diversity(){
  return(cv_errors_diversity.index_min()); 
}

// Computing the solutions over a grid for folds. Grid is either for the sparsity or the diverity (one of them is fixed)
void CV_CPGLIB::Compute_CV_Grid(arma::uvec & sample_ind, arma::uvec & fold_ind,
                                bool & diversity_search){ 
  
  if(!diversity_search){ // Search for optimal sparsity parameter
    
    // Initializing the sparsity CV Errors
    cv_errors_sparsity = arma::zeros(n_lambda_sparsity);
    
    // Looping over the folds
    # pragma omp parallel for num_threads(n_threads)
    for(arma::uword fold=0; fold<n_folds; fold++){ 
      
      // Get test and training samples
      arma::uvec test = arma::linspace<arma::uvec>(fold_ind[fold],
                                                   fold_ind[fold + 1] - 1,
                                                   fold_ind[fold + 1] - fold_ind[fold]);
      arma::uvec train = Set_Diff(sample_ind, test);
      
      // Initialization of the WEN objects (with the maximum value of lambda_sparsity_grid)
      CPGLIB CPGLIB_fold = CPGLIB(x.rows(train), y.elem(train),
                                  type, 
                                  G, 
                                  include_intercept,
                                  alpha_s, alpha_d,
                                  lambda_sparsity_grid[n_lambda_sparsity-1], lambda_diversity_opt,
                                  permutate_search,
                                  acceleration,
                                  tolerance, max_iter);
      
      // Looping over the different sparsity penalty parameters
      for(int sparsity_ind=n_lambda_sparsity-1; sparsity_ind>=0; sparsity_ind--){
        
        // Setting the lambda_sparsity value
        CPGLIB_fold.Set_Lambda_Sparsity(lambda_sparsity_grid[sparsity_ind]);
        // Computing the betas for the fold (new lambda_sparsity)
        CPGLIB_fold.Cycle_Groups();
        // Computing the deviance for the fold (new lambda_sparsity)
        Compute_CV_Deviance_Sparsity(sparsity_ind,
                                     x.rows(test), y.elem(test),
                                     CPGLIB_fold.Get_Intercept_Scaled(), CPGLIB_fold.Get_Coef_Scaled());
        
      } // End of loop over the sparsity parameter values
      
    } // End of loop over the folds
    
    // Storing the optimal sparsity parameters
    index_sparsity_opt = cv_errors_sparsity.index_min();
    lambda_sparsity_opt = lambda_sparsity_grid[index_sparsity_opt];
    cv_opt_new = arma::min(cv_errors_sparsity);
    
  } 
  else{
    
    // Computing the grid for the diversity parameter
    Compute_Lambda_Diversity_Grid();
    
    // Initializing the diversity CV Errors
    cv_errors_diversity = arma::zeros(n_lambda_diversity);
    
    // Looping over the folds
    for(arma::uword fold=0; fold<n_folds; fold++){ 
      
      // Get test and training samples
      arma::uvec test = arma::linspace<arma::uvec>(fold_ind[fold],
                                                   fold_ind[fold + 1] - 1,
                                                   fold_ind[fold + 1] - fold_ind[fold]);
      arma::uvec train = Set_Diff(sample_ind, test);
      
      // Initialization of the WEN objects (with the maximum value of lambda_sparsity_grid)
      CPGLIB CPGLIB_fold = CPGLIB(x.rows(train), y.elem(train),
                                  type, 
                                  G, 
                                  include_intercept,
                                  alpha_s, alpha_d,
                                  lambda_sparsity_opt, lambda_diversity_grid[n_lambda_diversity-1],
                                  permutate_search,
                                  acceleration,
                                  tolerance, max_iter);
      
      // Looping over the different diversity penalty parameters
      for(int diversity_ind = n_lambda_diversity-1; diversity_ind>=0; diversity_ind--){
        
        // Setting the lambda_sparsity value
        CPGLIB_fold.Set_Lambda_Diversity(lambda_diversity_grid[diversity_ind]);
        // Computing the betas for the fold (new lambda_sparsity)
        CPGLIB_fold.Cycle_Groups();
        // Computing the deviance for the fold (new lambda_sparsity)
        Compute_CV_Deviance_Diversity(diversity_ind,
                                      x.rows(test), y.elem(test),
                                      CPGLIB_fold.Get_Intercept_Scaled(), CPGLIB_fold.Get_Coef_Scaled());
        
      } // End of loop over the sparsity parameter values
      
    } // End of loop over the folds
    
    // Storing the optimal diversity parameters
    index_diversity_opt = cv_errors_diversity.index_min();
    lambda_diversity_opt = lambda_diversity_grid[index_diversity_opt];
    cv_opt_new = arma::min(cv_errors_diversity);
  }
}

// Computing the solutions over a grid for folds. Grid is either for the sparsity or the diverity (one of them is fixed)
void CV_CPGLIB::Compute_CV_Grid_Balanced(arma::uvec & sample_ind, arma::uvec & fold_ind,
                                         bool & diversity_search){ 
  
  if(!diversity_search){ // Search for optimal sparsity parameter
    
    // Initializing the sparsity CV Errors
    cv_errors_sparsity = arma::zeros(n_lambda_sparsity);
    
    // Looping over the folds
# pragma omp parallel for num_threads(n_threads)
    for(arma::uword fold=0; fold<n_folds; fold++){ 
      
      // Get test and training samples
      arma::uvec test = arma::linspace<arma::uvec>(fold_ind[fold],
                                                   fold_ind[fold + 1] - 1,
                                                   fold_ind[fold + 1] - fold_ind[fold]);
      arma::uvec train = Set_Diff(sample_ind, test);
      
      // Initialization of the WEN objects (with the maximum value of lambda_sparsity_grid)
      CPGLIB CPGLIB_fold = CPGLIB(x.rows(train), y.elem(train),
                                  type, 
                                  G, 
                                  include_intercept,
                                  alpha_s, alpha_d,
                                  lambda_sparsity_grid[0], lambda_diversity_opt,
                                  permutate_search,
                                  acceleration,
                                  tolerance, max_iter);
      
      // Looping over the different sparsity penalty parameters
      for(int sparsity_ind=0; sparsity_ind<n_lambda_sparsity; sparsity_ind++){
        
        // Setting the lambda_sparsity value
        CPGLIB_fold.Set_Lambda_Sparsity(lambda_sparsity_grid[sparsity_ind]);
        // Computing the betas for the fold (new lambda_sparsity)
        CPGLIB_fold.Cycle_Groups_Balanced();
        // Computing the deviance for the fold (new lambda_sparsity)
        Compute_CV_Deviance_Sparsity(sparsity_ind,
                                     x.rows(test), y.elem(test),
                                     CPGLIB_fold.Get_Intercept_Scaled(), CPGLIB_fold.Get_Coef_Scaled());
        
      } // End of loop over the sparsity parameter values
      
    } // End of loop over the folds
    
    // Storing the optimal sparsity parameters
    index_sparsity_opt = cv_errors_sparsity.index_min();
    lambda_sparsity_opt = lambda_sparsity_grid[index_sparsity_opt];
    cv_opt_new = arma::min(cv_errors_sparsity);
    
  } 
  else{
    
    // Computing the grid for the diversity parameter
    Compute_Lambda_Diversity_Grid();
    
    // Initializing the diversity CV Errors
    cv_errors_diversity = arma::zeros(n_lambda_diversity);
    
    // Looping over the folds
    for(arma::uword fold=0; fold<n_folds; fold++){ 
      
      // Get test and training samples
      arma::uvec test = arma::linspace<arma::uvec>(fold_ind[fold],
                                                   fold_ind[fold + 1] - 1,
                                                   fold_ind[fold + 1] - fold_ind[fold]);
      arma::uvec train = Set_Diff(sample_ind, test);
      
      // Initialization of the WEN objects (with the maximum value of lambda_sparsity_grid)
      CPGLIB CPGLIB_fold = CPGLIB(x.rows(train), y.elem(train),
                                  type, 
                                  G, 
                                  include_intercept,
                                  alpha_s, alpha_d,
                                  lambda_sparsity_opt, lambda_diversity_grid[0],
                                  permutate_search,
                                  acceleration,
                                  tolerance, max_iter);
      
      // Looping over the different diversity penalty parameters
      for(int diversity_ind = 0; diversity_ind < n_lambda_diversity; diversity_ind++){
        
        // Setting the lambda_sparsity value
        CPGLIB_fold.Set_Lambda_Diversity(lambda_diversity_grid[diversity_ind]);
        // Computing the betas for the fold (new lambda_sparsity)
        CPGLIB_fold.Cycle_Groups_Balanced();
        // Computing the deviance for the fold (new lambda_sparsity)
        Compute_CV_Deviance_Diversity(diversity_ind,
                                      x.rows(test), y.elem(test),
                                      CPGLIB_fold.Get_Intercept_Scaled(), CPGLIB_fold.Get_Coef_Scaled());
        
      } // End of loop over the sparsity parameter values
      
    } // End of loop over the folds
    
    // Storing the optimal diversity parameters
    index_diversity_opt = cv_errors_diversity.index_min();
    lambda_diversity_opt = lambda_diversity_grid[index_diversity_opt];
    cv_opt_new = arma::min(cv_errors_diversity);
  }
}


// Function for the initial CV Error with no diversity
void CV_CPGLIB::Get_CV_Sparsity_Initial(){  

  CV_ProxGrad initial_model = CV_ProxGrad(x, y,
                                          type, 
                                          include_intercept, 
                                          alpha_s,
                                          n_lambda_sparsity,
                                          acceleration,
                                          tolerance, max_iter,
                                          n_folds,
                                          n_threads);

  // Computation of coefficients
  initial_model.Compute_CV_Betas();

  // Storing the optimal sparsity parameters
  cv_errors_sparsity = initial_model.Get_CV_Error_Sparsity();
  index_sparsity_opt = cv_errors_sparsity.index_min();
  lambda_sparsity_opt = initial_model.Get_Lambda_Sparsity_Grid()[(initial_model.Get_CV_Error_Sparsity()).index_min()]; 
  cv_opt_new = arma::min(cv_errors_sparsity);
}

// Coordinate descent algorithms for coefficients
void CV_CPGLIB::Compute_CV_Betas(){
  
  // Creating indices for the folds of the data
  arma::uvec sample_ind = arma::linspace<arma::uvec>(0, n-1, n);
  arma::uvec fold_ind = arma::linspace<arma::uvec>(0, n, n_folds+1);

  // Initial iteration with no diversity
  Get_CV_Sparsity_Initial();
  double cv_opt_no_diversity = cv_opt_old = cv_opt_new;

  // Variables to store the old penalty parameters
  double lambda_sparsity_opt_old, lambda_diversity_opt_old;
  
  // // Initial cycle
  bool diversity_search = true;
  if(balanced_cycling)
    Compute_CV_Grid_Balanced(sample_ind, fold_ind, diversity_search);
  else
    Compute_CV_Grid(sample_ind, fold_ind, diversity_search);
  
  // // Print iteration data to console (commented out for package)
  // std::cout << "Iteration 1: " << std::endl;
  // std::cout << "cv_opt_old: " << cv_opt_old << std::endl;
  // std::cout << "cv_opt_new: " << cv_opt_new << std::endl;
  // std::cout << "lambda_sparsity_opt: " << lambda_sparsity_opt << std::endl;
  // std::cout << "lambda_diversity_opt: " << lambda_diversity_opt << std::endl << std::endl;

  // Computing the solutions until the optimal is no longer a significant improvement
  arma::uword cv_iterations = 1;
  do{
    
    // Storing the old parameters
    cv_opt_old = cv_opt_new;
    lambda_sparsity_opt_old = lambda_sparsity_opt;
    lambda_diversity_opt_old = lambda_diversity_opt;
    
    // New search over the penalty parameters
    diversity_search = !diversity_search;
    if(balanced_cycling)
      Compute_CV_Grid_Balanced(sample_ind, fold_ind, diversity_search);
    else
      Compute_CV_Grid(sample_ind, fold_ind, diversity_search);
    
    cv_iterations++;
    
    // // Print iteration data to console (commented out for package)
    // std::cout << "Iteration: " << cv_iterations  << std::endl;
    // std::cout << "cv_opt_old: " << cv_opt_old << std::endl;
    // std::cout << "cv_opt_new: " << cv_opt_new << std::endl;
    // std::cout << "lambda_sparsity_opt: " << lambda_sparsity_opt << std::endl;
    // std::cout << "lambda_diversity_opt: " << lambda_diversity_opt << std::endl << std::endl;
    
    // Conditions for breaking out of search for optimal penalty parameters
    if(cv_opt_new > cv_opt_old || 
       (!diversity_search && lambda_sparsity_opt==lambda_sparsity_opt_old) || 
       (diversity_search && lambda_diversity_opt==lambda_diversity_opt_old)){
      
      lambda_sparsity_opt = lambda_sparsity_opt_old;
      lambda_diversity_opt = lambda_diversity_opt_old;
      break;
    }
    
  } while (std::fabs(cv_opt_new-cv_opt_old)>CV_ITERATIONS_TOLERANCE && cv_iterations<CV_ITERATIONS_MAX);
  
  
  // Computing the parameters for the full data
  CPGLIB CPGLIB_full = CPGLIB(x, y,
                              type, 
                              G, 
                              include_intercept,
                              alpha_s, alpha_d,
                              lambda_sparsity_grid[n_lambda_sparsity-1], lambda_diversity_opt,
                              permutate_search,
                              acceleration,
                              tolerance, max_iter);
  
  // Initial parameters computation
  if(balanced_cycling)
    CPGLIB_full.Cycle_Groups_Balanced_Grid();
  else
    CPGLIB_full.Cycle_Groups_Grid();
  intercepts.col(n_lambda_sparsity-1) =  CPGLIB_full.Get_Intercept_Scaled();
  betas.slice(n_lambda_sparsity-1) = CPGLIB_full.Get_Coef_Scaled();

  if(balanced_cycling){
    
    // Looping over the different sparsity penalty parameters
    for(int sparsity_ind=n_lambda_sparsity-1; sparsity_ind>=0; sparsity_ind--){
      
      // Setting the lambda_sparsity value
      CPGLIB_full.Set_Lambda_Sparsity(lambda_sparsity_grid[sparsity_ind]);
      // Computing the betas for the fold (new lambda_sparsity)
      CPGLIB_full.Cycle_Groups_Balanced();
      // Storing the full data models
      intercepts.col(sparsity_ind) =  CPGLIB_full.Get_Intercept_Scaled();
      betas.slice(sparsity_ind) = CPGLIB_full.Get_Coef_Scaled();
      
    } // End of loop over the sparsity parameter values
  }
  else{
    
    // Looping over the different sparsity penalty parameters
    for(arma::uword sparsity_ind = 1; sparsity_ind < n_lambda_sparsity; sparsity_ind++){
      
      // Setting the lambda_sparsity value
      CPGLIB_full.Set_Lambda_Sparsity(lambda_sparsity_grid[sparsity_ind]);
      // Computing the betas for the fold (new lambda_sparsity)
      CPGLIB_full.Cycle_Groups();
      // Storing the full data models
      intercepts.col(sparsity_ind) =  CPGLIB_full.Get_Intercept_Scaled();
      betas.slice(sparsity_ind) = CPGLIB_full.Get_Coef_Scaled();
      
    } // End of loop over the sparsity parameter values
  }
  
}

// Coordinate descent algorithms for coefficients (Full Diveristy)
void CV_CPGLIB::Compute_CV_Betas_Full_Diversity(){ 

  // Creating indices for the folds of the data
  arma::uvec sample_ind = arma::linspace<arma::uvec>(0, n-1, n);
  arma::uvec fold_ind = arma::linspace<arma::uvec>(0, n, n_folds+1);

  // Computing the solutions for the folds for all sparsity levels
  lambda_sparsity_opt = eps_sparsity;
  lambda_diversity_opt = Get_Lambda_Diversity_Max();
  bool diversity_search = false;
  Compute_CV_Grid(sample_ind, fold_ind, diversity_search);

  // Computing the parameters for the full diversity for optimal sparsity parameter
  CPGLIB CPGLIB_full = CPGLIB(x, y,
                              type,
                              G, 
                              include_intercept,
                              alpha_s, alpha_d,
                              lambda_sparsity_grid[lambda_sparsity_grid.n_elem-1], lambda_diversity_opt,
                              permutate_search,
                              acceleration,
                              tolerance, max_iter);
  
  if(balanced_cycling){
    
    // Looping over the different sparsity penalty parameters
    for(int sparsity_ind=n_lambda_sparsity-1; sparsity_ind>=0; sparsity_ind--){
      
      // Setting the lambda_sparsity value
      CPGLIB_full.Set_Lambda_Sparsity(lambda_sparsity_grid[sparsity_ind]);
      // Computing the betas for the fold (new lambda_sparsity)
      CPGLIB_full.Cycle_Groups_Balanced();
      // Storing the full data models
      intercepts.col(sparsity_ind) =  CPGLIB_full.Get_Intercept_Scaled();
      betas.slice(sparsity_ind) = CPGLIB_full.Get_Coef_Scaled();
      
    } // End of loop over the sparsity parameter values
  }
  else{
    
    // Looping over the different sparsity penalty parameters
    for(int sparsity_ind=n_lambda_sparsity-1; sparsity_ind>=0; sparsity_ind--){ 
  
      // Setting the lambda_sparsity value
      CPGLIB_full.Set_Lambda_Sparsity(lambda_sparsity_grid[sparsity_ind]);
      // Computing the betas for the fold (new lambda_sparsity)
      CPGLIB_full.Cycle_Groups();
      // Storing the full data models
      intercepts.col(sparsity_ind) =  CPGLIB_full.Get_Intercept_Scaled();
      betas.slice(sparsity_ind) = CPGLIB_full.Get_Coef_Scaled();
  
    } // End of loop over the sparsity parameter values
  }

}

// Class destructor
CV_CPGLIB::~CV_CPGLIB(){ 
}

/*
* ________________________________________________
* Static Functions - Deviance
* ________________________________________________
*/

// Static FUnctions - Linear Deviance
double CV_CPGLIB::Linear_Deviance(arma::mat x, arma::vec y,
                                  arma::vec intercept, arma::mat betas){
  
  arma::vec linear_fit = arma::mean(intercept) + x*arma::mean(betas,1);
  return(arma::accu(arma::square(linear_fit - y))/2);
}

// Static FUnctions - Logistic Deviance
double CV_CPGLIB::Logistic_Deviance(arma::mat x, arma::vec y, 
                                    arma::vec intercept, arma::mat betas){
  
  arma::vec linear_fit = arma::mean(intercept) + x*arma::mean(betas,1);
  return(2*arma::accu(arma::log(1 + arma::exp(linear_fit)) - linear_fit % y));
}

// Static FUnctions - Gamma Deviance
double CV_CPGLIB::Gamma_Deviance(arma::mat x, arma::vec y, 
                                 arma::vec intercept, arma::mat betas){
  
  arma::vec linear_fit = arma::mean(intercept) + x*arma::mean(betas,1);
  return(2*arma::accu(linear_fit + arma::exp(-linear_fit) % y));
}

// Static FUnctions - Poisson Deviance
double CV_CPGLIB::Poisson_Deviance(arma::mat x, arma::vec y, 
                                   arma::vec intercept, arma::mat betas){
  
  arma::vec linear_fit = arma::mean(intercept) + x*arma::mean(betas,1);
  return(2*arma::accu(arma::exp(linear_fit) - linear_fit % y));
}






