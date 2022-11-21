/*
* ===========================================================
* File Type: HPP
* File Name: CPGLIB.hpp
* Package Name: CPGLIB
*
* Created by Anthony-A. Christidis.
* Copyright (c) Anthony-A. Christidis. All rights reserved.
* ===========================================================
*/

#ifndef CPGLIB_hpp
#define CPGLIB_hpp

#include <RcppArmadillo.h>

#include "config.h" 

class CPGLIB{
  
private:
  
  // Variables supplied by user
  arma::mat x;
  arma::vec y; 
  arma::uword type;
  arma::uword G;
  arma::uword include_intercept;
  double alpha_s;
  double alpha_d;
  double lambda_sparsity;
  double lambda_diversity;
  double lambda_diversity_max;
  arma::vec lambda_sparsity_grid;
  arma::vec lambda_diversity_grid;
  arma::vec prox_threshold_step;
  arma::vec prox_scale_step;
  double tolerance;
  arma::uword max_iter;
  
  // Variables created inside class
  arma::rowvec mu_x;
  arma::rowvec sd_x;
  arma::mat x_std_aug;
  double mu_y;
  double sd_y;
  arma::uword n; // Number of samples
  arma::uword p; // Number of variables (including intercept term)
  arma::vec intercept_scaled;
  arma::mat betas;
  arma::mat new_betas;
  arma::mat betas_scaled;
  arma::vec grad_vector; // Vector for the gradient update
  arma::vec grad_step; // Vector for the size of the gradient update
  arma::vec active_set;
  arma::vec new_active_set;
  arma::mat expected_val; // Vector for the expected values
  arma::uword iter_count;
  double step_size;
  
  // Function to initial the object characteristics
  void Initialize();
  
  // Gradient step size computation
  void (*Compute_Gradient_Step)(arma::vec & grad_step, arma::vec & grad_vector);
  
  // Gradient proposals iterations
  void (*Proposal_Iteration)(double & t_prev, double & t_next, 
        arma::mat & betas, arma::mat & new_betas, arma::vec & proposal,
        arma::uword & group);
  
  // Functions for the computation of (negative) log-likelihoods
  double (*Compute_Likelihood)(arma::mat & x, arma::vec & y, 
          arma::mat & betas, 
          arma::uword & group);
  // Functions for the computation of coefficients
  void (*Compute_Gradient)(arma::mat & x, arma::vec & y, 
        arma::mat & betas, arma::vec & grad_vector);
  // Soft-threshold operator
  arma::vec Soft(arma::vec & u, 
                 arma::vec & threshold, arma::vec & scale);
  
  // Functions for the computation of expected values
  void (*Compute_Expected)(arma::mat & x, 
        arma::mat & betas, arma::vec & expected_val,
        arma::uword & group);
  
  // Functions to compute the diversity weights (L1 and L2)
  arma::vec Beta_Weights_Abs(arma::uword & group);
  arma::vec Beta_Weights_Sq(arma::uword & group);
  
  // Function to copmute the diversity penalty (fixed group)
  double Sparsity_Penalty(arma::uword & group);
  double Sparsity_Penalty_New(arma::uword & group);
  
  // Sparsity and diversity penalties (objective function)
  double Sparsity_Penalty();
  double Diversity_Penalty();
  
  // Function to compare the active sets for convergence
  void Update_Active_Set();
  bool Compare_Active_Set();
  
  // Method to get the grid of lambda_sparsity
  void Compute_Lambda_Sparsity_Grid();
  // Method to get diversity penalty parameter that kills all interactions
  void Compute_Lambda_Diversity_Max();
  // Method to get the grid of lambda_diversity
  void Compute_Lambda_Diversity_Grid();
  
  // Function that checks if there are interactions between groups in the matrix of betas
  bool Check_Interactions_Beta(arma::mat beta);
  // Function to returns a vector with ones corresponding to the betas that have interactions.
  arma::uvec Check_Interactions(arma::cube & betas);

public:
  
  // Constructor - with data
  CPGLIB(arma::mat x, arma::vec y,
         arma::uword & type,
         arma::uword & G, 
         arma::uword & include_intercept,
         double alpha_s, double alpha_d,
         double lambda_sparsity, double lambda_diversity,
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
  // Method to set alpha_d to new value and return current alpha_d
  void Set_Alpha_Diversity(double alpha_d);
  double Get_Alpha_Diversity();
  
  // Method to set lambda_sparsity to new value and return current lambda_sparsity
  void Set_Lambda_Sparsity(double lambda_sparsity);
  double Get_Lambda_Sparsity();
  // Method to set lambda_diversity to new value and return current lambda_diversity
  void Set_Lambda_Diversity(double lambda_diversity);
  double Get_Lambda_Diversity();
  
  // Function to set coefficients
  void Set_Betas(arma::uword group, arma::vec set_betas);

  // Function to initialize coefficients - no diversity
  void Initialize_Betas_No_Diversity();
  
  // Function to return expected values
  arma::vec Get_Expected();
  
  // Function to compute the objective function value (fixed group)
  double Get_Objective_Value(arma::uword & group);
  double Get_Objective_Value_New(arma::uword & group);
  
  // Function to compute the global objective function value
  double Get_Objective_Value();
  
  // Function to compute coefficients
  void Coef_Update(arma::uword & group);
  void Compute_Coef();

  // Function to return the number of iterations for convergence
  arma::uword Get_Iter();
  
  // Function to scale back coefficients to original scale
  void Scale_Coefficients();
  void Scale_Intercept();
  
  // Methods to return coefficients
  arma::mat Get_Coef();
  arma::mat Get_Coef_Scaled();
  arma::rowvec Get_Intercept();
  arma::vec Get_Intercept_Scaled();
  
  // ---------------------------------
  // Static Functions - GLM Dependent
  // ---------------------------------
  
  // Static FUnctions - (Negative) Log-Likelihoods Computation
  static double Linear_Likelihood(arma::mat & x, arma::vec & y, 
                                  arma::mat & betas, 
                                  arma::uword & group);
  static double Logistic_Likelihood(arma::mat & x, arma::vec & y, 
                                    arma::mat & betas, 
                                    arma::uword & group);
  
  // Static FUnctions - Gradients Computation
  static void Linear_Gradient(arma::mat & x, arma::vec & y, 
                              arma::mat & betas, arma::vec & grad_vector);
  static void Logistic_Gradient(arma::mat & x, arma::vec & y, 
                                arma::mat & betas, arma::vec & grad_vector);

  // Static FUnctions - Expected Values Computation
  static void Linear_Expected(arma::mat & x, arma::mat & betas, 
                              arma::vec & expected_val,
                              arma::uword & group);
  static void Logistic_Expected(arma::mat & x, arma::mat & betas, 
                                arma::vec & expected_val,
                                arma::uword & group);
  
  // Destructor
  ~CPGLIB();
};

#endif // CPGLIB_hpp
