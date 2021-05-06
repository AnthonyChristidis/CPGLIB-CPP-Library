/*
* ===========================================================
* File Type: HPP
* File Name: ProxGrad.hpp
* Package Name: CPGLIB
*
* Created by Anthony-A. Christidis.
* Copyright © Anthony-A. Christidis. All rights reserved.
* ===========================================================
*/

#ifndef ProxGrad_hpp
#define ProxGrad_hpp

#include <RcppArmadillo.h>

#include "config.h"

class ProxGrad{
  
private:
  
  // Variables supplied by user
  arma::mat x;
  arma::vec y; 
  arma::uword type;
  arma::uword G;
  arma::uword include_intercept;
  double alpha_s;
  // double alpha_d;
  double lambda_sparsity;
  // double lambda_diversity;
  arma::uword acceleration;
  double tolerance;
  arma::uword max_iter;
  
  // Variables created inside class
  arma::rowvec mu_x;
  arma::rowvec sd_x;
  arma::mat x_std_aug;
  // double sd_y;
  // arma::vec y_std;
  arma::uword n; // Number of samples
  arma::uword p; // Number of variables (including intercept term)
  double intercept_scaled;
  arma::vec betas;
  arma::vec new_betas;
  arma::vec betas_scaled;
  arma::vec grad_vector; // Vector for the gradient update
  arma::vec grad_step; // Vector for the size of the gradient update
  arma::vec active_set;
  arma::vec new_active_set;
  arma::vec expected_val; // Vector for the expected values
  arma::uword iter_count;

  // Function to initial the object characteristics
  void Initialize();
  
  // Gradient step size computation
  void (*Compute_Gradient_Step)(arma::vec & grad_step, arma::vec & grad_vector);
  
  // Gradient proposals iterations
  void (*Proposal_Iteration)(double & t_prev, double & t_next, arma::vec & betas, arma::vec & new_betas, arma::vec & proposal);
  
  // Functions for the computation of (negative) log-likelihoods
  double (*Compute_Likelihood)(arma::mat & x, arma::vec & y, 
          arma::vec & betas);
  // Functions for the computation of coefficients
  void (*Compute_Gradient)(arma::mat & x, arma::vec & y, 
        arma::vec & betas, arma::vec & grad_vector);
  // Soft-threshold operator
  arma::vec Soft(arma::vec & u, arma::vec & threshold, arma::vec & scale);

  // Functions for the computation of expected values
  void (*Compute_Expected)(arma::mat & x, arma::vec & betas, arma::vec & expected_val);
  
  // // Functions to compute the diversity weights (L1 and L2)
  // arma::vec Beta_Weights_Abs(arma::uword & group);
  // arma::vec Beta_Weights_Sq(arma::uword & group);
  
  // Function to copmute the diversity penalty
  double Sparsity_Penalty();
  double Sparsity_Penalty_New();
  // double Diversity_Penalty();
  
  // Function to compare the active sets for convergence
  void Update_Active_Set();
  bool Compare_Active_Set();

public:
  
  // Constructor - with data
  ProxGrad(arma::mat x, arma::vec y,
           arma::uword & type,
           arma::uword & include_intercept,
           double alpha_s,
           double lambda_sparsity, 
           arma::uword & acceleration,
           double tolerance, arma::uword max_iter);
  
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
  
  // Method to set lambda_sparsity to new value and return current lambda_sparsity
  void Set_Lambda_Sparsity(double lambda_sparsity);
  double Get_Lambda_Sparsity();
  // // Method to set lambda_diversity to new value and return current lambda_diversity
  // void Set_Lambda_Diversity(double lambda_diversity);
  // double Get_Lambda_Diversity();
  
  // Function to return expected values
  arma::vec Get_Expected();
  
  // Function to compute the objective value
  double Get_Objective_Value();
  double Get_Objective_Value_New();
  
  // Function to compute coefficients
  void Compute_Coef();
  
  // Function to return the number of iterations for convergence
  arma::uword Get_Iter();
  
  // Function to scale back coefficients to original scale
  void Scale_Coefficients();
  void Scale_Intercept();
  
  // Methods to return coefficients
  arma::vec Get_Coef();
  arma::vec Get_Coef_Scaled();
  double Get_Intercept();
  double Get_Intercept_Scaled();
  
  // ----------------------------------
  // Static Functions - Gradient Steps
  // ----------------------------------
  
  static void AdaGrad(arma::vec & grad_step, arma::vec & grad_vector);

  // --------------------------------------
  // Static Functions - Descent Iterations
  // --------------------------------------
  
  static void ISTA(double & t_prev, double & t_next, 
                   arma::vec & betas, arma::vec & new_betas, arma::vec & proposal);
  static void FISTA(double & t_prev, double & t_next, 
                    arma::vec & betas, arma::vec & new_betas, arma::vec & proposal);
  
  // ---------------------------------
  // Static Functions - GLM Dependent
  // ---------------------------------
  
  // Static FUnctions - (Negative) Log-Likelihoods Computation
  static double Linear_Likelihood(arma::mat & x, arma::vec & y, 
                                  arma::vec & betas);
  static double Logistic_Likelihood(arma::mat & x, arma::vec & y, 
                                    arma::vec & betas);
  static double Gamma_Likelihood(arma::mat & x, arma::vec & y, 
                                 arma::vec & betas);
  static double Poisson_Likelihood(arma::mat & x, arma::vec & y, 
                                   arma::vec & betas);
  
  // Static FUnctions - Gradients Computation
  static void Linear_Gradient(arma::mat & x, arma::vec & y, 
                              arma::vec & betas, arma::vec & grad_vector);
  static void Logistic_Gradient(arma::mat & x, arma::vec & y, 
                                arma::vec & betas, arma::vec & grad_vector);
  static void Gamma_Gradient(arma::mat & x, arma::vec & y, 
                             arma::vec & betas, arma::vec & grad_vector);
  static void Poisson_Gradient(arma::mat & x, arma::vec & y, 
                               arma::vec & betas, arma::vec & grad_vector);
  
  // Static FUnctions - Expected Values Computation
  static void Linear_Expected(arma::mat & x, arma::vec & betas, 
                              arma::vec & expected_val);
  static void Logistic_Expected(arma::mat & x, arma::vec & betas, 
                                arma::vec & expected_val);
  static void Gamma_Expected(arma::mat & x, arma::vec & betas, 
                             arma::vec & expected_val);
  static void Poisson_Expected(arma::mat & x, arma::vec & betas, 
                               arma::vec & expected_val);
  
  // Destructor
  ~ProxGrad();
};

#endif // ProxGrad_hpp
