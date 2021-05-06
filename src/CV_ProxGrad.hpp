/*
* ===========================================================
* File Type: HPP
* File Name: CV_ProxGrad.hpp
* Package Name: CPGLIB
*
* Created by Anthony-A. Christidis.
* Copyright � Anthony-A. Christidis. All rights reserved.
* ===========================================================
*/

#ifndef CV_ProxGrad_hpp
#define CV_ProxGrad_hpp

#include <RcppArmadillo.h>

#include "config.h"

class CV_ProxGrad{
  
private:
  
  // Variables supplied by user
  arma::mat x;
  arma::vec y; 
  arma::uword type;
  arma::uword G;
  arma::uword include_intercept;
  double alpha_s;
  // double alpha_d;
  arma::uword n_lambda_sparsity;
  // arma::uword n_lambda_diversity;
  arma::uword acceleration;
  double tolerance;
  arma::uword max_iter;
  arma::uword n_folds;
  
  // Variables created inside class
  arma::rowvec mu_x;
  arma::rowvec sd_x;
  arma::mat x_std_aug;
  double mu_y;
  arma::uword n; // Number of samples
  arma::uword p; // Number of variables (does not include intercept term)
  arma::vec lambda_sparsity_grid;
  // arma::vec lambda_diversity_grid;
  double eps;
  arma::vec intercepts;
  arma::mat betas;
  arma::vec cv_errors_sparsity;
  // arma::vec cv_errors_diversity;
  double cv_opt_old;
  double cv_opt_new;
  arma::uword index_sparsity_opt;
  double lambda_sparsity_opt;
  // arma::uword index_diversity_opt;
  // double lambda_diversity_opt;
  arma::uword n_threads;
  
  // Function to initial the object characteristics
  void Initialize();
  
  // Method to get the grid of lambda_sparsity
  void Compute_Lambda_Sparsity_Grid();
  // // Method to get diversity penalty parameter that kills all interactions
  // double Get_Lambda_Diversity_Max();
  // // Method to get the grid of lambda_diversity
  // void Compute_Lambda_Diversity_Grid();
  
  // // Function that checks if there are interactions between groups in the matrix of betas
  // bool Check_Interactions_Beta(arma::mat beta);
  // // Function to returns a vector with ones corresponding to the betas that have interactions.
  // arma::uvec Check_Interactions(arma::cube & betas);
  
  // Private function to create the folds
  arma::uvec Set_Diff(const arma::uvec & big, const arma::uvec & small);
  
  // Private function to compute the CV-MSPE over the folds
  void Compute_CV_Deviance_Sparsity(arma::uword & sparsity_ind,
                                    arma::mat x_test, arma::vec y_test,
                                    double intercept, arma::vec betas);
  // void Compute_CV_Deviance_Diversity(int diversity_ind,
  //                                    arma::mat x_test, arma::vec y_test,
  //                                    arma::vec intercept, arma::mat betas);
  double (*Compute_Deviance)(arma::mat & x, arma::vec & y,
          double & intercept, arma::vec & betas);
  
public:
  
  // Constructor - with data
  CV_ProxGrad(arma::mat & x, arma::vec & y,
              arma::uword & type,
              arma::uword & include_intercept,
              double & alpha_s,
              arma::uword & n_lambda_sparsity, 
              arma::uword & acceleration,
              double & tolerance, arma::uword & max_iter,
              arma::uword & n_folds,
              arma::uword & n_threads);
  
  // Functions to set new data
  void Set_X(arma::mat & x);
  void Set_Y(arma::vec & y);
  
  // Functions to set maximum number of iterations and tolerance
  void Set_Max_Iter(arma::uword & max_iter);
  void Set_Tolerance(double & tolerance);
  
  // Method to set alpha_s to new value and return current alpha_s
  void Set_Alpha_Sparsity(double alpha_s);
  double Get_Alpha_Sparsity();
  // // Method to set alpha_d to new value and return current alpha_d
  // void Set_Alpha_Diversity(double alpha_s);
  // double Get_Alpha_Diversity();
  
  // Method to get the grid of lambda sparsity
  arma::vec Get_Lambda_Sparsity_Grid();
  // // Method to get the grid of lambda diversity
  // arma::vec Get_Lambda_Diversity_Grid();
  
  // Cross-validation - Sparsity
  arma::vec Get_CV_Error_Sparsity();
  // // Cross-validation - Diversity
  // arma::vec Get_CV_Error_Diversity();
  
  // Optimal penalty parameter - Sparsity
  double Get_lambda_sparsity_opt();
  // // Optimal penalty parameter - Diversity
  // double Get_lambda_diversity_opt();
  
  // Methods to return coefficients
  arma::mat Get_Coef();
  arma::vec Get_Intercept();
  
  // Optimal sparsity parameter
  arma::uword Get_Optimal_Index_Sparsity();
  // // Optimal diversity parameter
  // arma::uword Get_Optimal_Index_Diversity();
  
  // Computing the solutions over a grid for folds. Grid is either for the sparsity or the diverity (one of them is fixed)
  void Compute_CV_Grid(arma::uvec & sample_ind, arma::uvec & fold_ind);

  // Coordinate descent algorithms for coefficients
  void Compute_CV_Betas();

  // -----------------------------
  // Static Functions - Deviance
  // -----------------------------
  
  static double Linear_Deviance(arma::mat & x, arma::vec & y,
                                double & intercept, arma::vec & betas);
  static double Logistic_Deviance(arma::mat & x, arma::vec & y,
                                  double & intercept, arma::vec & betas);
  static double Gamma_Deviance(arma::mat & x, arma::vec & y,
                               double & intercept, arma::vec & betas);
  static double Poisson_Deviance(arma::mat & x, arma::vec & y,
                                 double & intercept, arma::vec & betas);
  
  // Destructor
  ~CV_ProxGrad();
};

#endif // ProxGrad_hpp
