/*
* ===========================================================
* File Type: CPP
* File Name: ProxGrad.cpp
* Package Name: CPGLIB
*
* Created by Anthony-A. Christidis.
* Copyright © Anthony-A. Christidis. All rights reserved.
* ===========================================================
*/

#include "ProxGrad.hpp"

#include <RcppArmadillo.h>

#include "config.h" 

#include <math.h>

// Constants - COMPUTATION 
const static double LS_CONST_PARAMETER = 0.25;

// Constructor for ProxGrad
ProxGrad::ProxGrad(arma::mat x, arma::vec y,
                   arma::uword & type,
                   arma::uword & include_intercept,
                   double alpha_s, 
                   double lambda_sparsity,
                   arma::uword & acceleration,
                   double tolerance, arma::uword max_iter):
  x(x), y(y),
  type(type),
  include_intercept(include_intercept),
  alpha_s(alpha_s), 
  lambda_sparsity(lambda_sparsity),
  acceleration(acceleration),
  tolerance(tolerance), max_iter(max_iter){
  
  // Initializing the object
  Initialize();
}
void ProxGrad::Initialize(){
  
  // Standardization of design matrix
  mu_x = arma::mean(x);
  sd_x = arma::stddev(x, 1);
  x.each_row() -= mu_x;
  x.each_row() /= sd_x;

  // Setting the parameters
  n = x.n_rows;
  p = x.n_cols + 1;
  betas = arma::zeros(p);
  new_betas = arma::zeros(p);
  grad_vector = arma::zeros(p);
  active_set = arma::zeros(p);
  new_active_set = arma::zeros(p);
  
  // Augmented matrix for optimization
  if(include_intercept)
    x_std_aug = arma::join_rows(arma::ones(n, 1), x);
  else
    x_std_aug = arma::join_rows(arma::zeros(n, 1), x);

  // Setting initial values and function pointers for expected values and weights
  if(type==1){ // Linear Model
    
    Compute_Likelihood = &ProxGrad::Linear_Likelihood;
    Compute_Gradient = &ProxGrad::Linear_Gradient;
    Compute_Expected = &ProxGrad::Linear_Expected;
    
    if(include_intercept)
      betas[0] = arma::mean(y);
  }
    
  else if(type==2){ // Logistic Regression
    
    Compute_Likelihood = &ProxGrad::Logistic_Likelihood;
    Compute_Gradient = &ProxGrad::Logistic_Gradient;
    Compute_Expected = &ProxGrad::Logistic_Expected;
    
    if(include_intercept)
      betas[0] = std::log(arma::mean(y)/(1-arma::mean(y)));
  }
  
  else if(type==3){ // Gamma GLM
    
    Compute_Likelihood = &ProxGrad::Gamma_Likelihood;
    Compute_Gradient = &ProxGrad::Gamma_Gradient;
    Compute_Expected = &ProxGrad::Gamma_Expected;
    if(include_intercept)
      betas[0] = std::log(arma::mean(y));
  }
    
  else if(type==4){ // Poisson GLM
    
    Compute_Likelihood = &ProxGrad::Poisson_Likelihood;
    Compute_Gradient = &ProxGrad::Poisson_Gradient;
    Compute_Expected = &ProxGrad::Poisson_Expected;
    
    if(include_intercept)
      betas[0] = std::log(arma::mean(y));
  }
  
  // Gradient step sizes options
  Compute_Gradient_Step = &ProxGrad::AdaGrad;
  
  // Gradient proposals
  if(acceleration)
    Proposal_Iteration = &ProxGrad::FISTA;
  else 
    Proposal_Iteration = &ProxGrad::ISTA;
  
}

// Iterative Soft function
arma::vec ProxGrad::Soft(arma::vec & u, arma::vec & threshold, arma::vec & scale){ 
  
  grad_vector = u - threshold;
  grad_vector(arma::find(grad_vector < 0)).zeros();
  return((arma::sign(u) % grad_vector)/(1 + scale));
}

// Function to compute the sparsity penalty
double ProxGrad::Sparsity_Penalty(){ 
  
  return(lambda_sparsity*((1-alpha_s)*0.5*(arma::accu(arma::pow(betas, 2)) - std::pow(betas[0], 2)) + 
         alpha_s*(arma::accu(arma::abs(betas)) - std::abs(betas[0]))));
}
double ProxGrad::Sparsity_Penalty_New(){ 
  
  return(lambda_sparsity*((1-alpha_s)*0.5*(arma::accu(arma::pow(new_betas, 2)) - std::pow(new_betas[0], 2)) + 
         alpha_s*(arma::accu(arma::abs(new_betas)) - std::abs(new_betas[0]))));
}

// Function to return the active variables
void ProxGrad::Update_Active_Set(){ 
  
  active_set.ones();
  active_set(arma::find(new_betas == 0)).zeros();
  new_active_set.ones();
  new_active_set(arma::find(new_betas == 0)).zeros();
}
bool ProxGrad::Compare_Active_Set(){
  
  return(arma::accu(arma::abs(active_set - new_active_set)) == 0);
}

// Functions to set new data
void ProxGrad::Set_X(arma::mat & x){
  
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
void ProxGrad::Set_Y(arma::vec & y){
  this->y = y;
}

// Functions to set maximum number of iterations and tolerance
void ProxGrad::Set_Max_Iter(arma::uword & max_iter){
  this->max_iter = max_iter;
}
void ProxGrad::Set_Tolerance(double & tolerance){
  this->tolerance = tolerance;
}

// Functions to set and get alpha_s
void ProxGrad::Set_Alpha_Sparsity(double alpha_s){
  this->alpha_s = alpha_s;
}
double ProxGrad::Get_Alpha_Sparsity(){
  return(this->alpha_s);
}

// Functions to set and get lambda_sparsity
void ProxGrad::Set_Lambda_Sparsity(double lambda_sparsity){
  this->lambda_sparsity = lambda_sparsity;
}
double ProxGrad::Get_Lambda_Sparsity(){
  return(lambda_sparsity);
}

// Function to return expected values
arma::vec ProxGrad::Get_Expected(){
  return(expected_val);
}

// Function to compute the objective value
double ProxGrad::Get_Objective_Value(){
  
  return(Compute_Likelihood(x_std_aug, y, betas) + Sparsity_Penalty());
}
double ProxGrad::Get_Objective_Value_New(){
  
  return(Compute_Likelihood(x_std_aug, y, new_betas) + Sparsity_Penalty_New());
}

// Function to compute coefficients
void ProxGrad::Compute_Coef(){
  
  // Variables for proximal gradient descent
  arma::vec proposal = betas;
  iter_count = 0;
  arma::vec prox_vector = arma::ones(p);
  arma::vec prox_threshold, prox_scale;
  
  // Variables for Nesterov acceleration
  double t_prev = 1, t_next = 1;
  
  // Initial gradient computation
  Compute_Gradient(x_std_aug, y, proposal, grad_vector);
  Compute_Gradient_Step(grad_step, grad_vector);

  // Proximal gradient descent iterations
  do{
    
    betas = new_betas;
    Compute_Gradient(x_std_aug, y, proposal, grad_vector);
    Compute_Gradient_Step(grad_step, grad_vector);
    
    // Line search for (local) descent with acceleration
    do{
      
      // Proximal gradient descent step
      prox_vector = proposal - grad_step % grad_vector;
      prox_threshold = grad_step*lambda_sparsity*alpha_s;
      prox_scale = grad_step*lambda_sparsity*(1-alpha_s);
      new_betas = Soft(prox_vector, prox_threshold, prox_scale);
      new_betas[0] = prox_vector[0]; // Adjustment for intercept term (no shrinkage)
      
      // Line search adjustment
      grad_step = grad_step*LS_CONST_PARAMETER;

    } while (Get_Objective_Value_New() > Get_Objective_Value());
      
    // (!) The above conditional can be replaced by the quadratic approximation 
      
    // Iteration proposal
    Proposal_Iteration(t_prev, t_next, 
                       betas, new_betas, proposal);

    // Iteration for algorithm
    iter_count++;

  } while ((iter_count <= max_iter) && (arma::max(arma::square(new_betas - betas)) > tolerance));
  
  // Final coefficients
  betas = new_betas;

  // Scaling coefficients
  Scale_Coefficients();
  Scale_Intercept();
}

// Function to return the number of iterations for convergence
arma::uword ProxGrad::Get_Iter(){
  return(iter_count);
}

// Function to scale back coefficients to original scale
void ProxGrad::Scale_Coefficients(){
  
  betas_scaled = betas.elem(arma::linspace<arma::uvec>(1, p-1, p-1));
  betas_scaled = betas_scaled % (1/sd_x.t());
}
void ProxGrad::Scale_Intercept(){
  
  intercept_scaled = ((include_intercept) ? 1 : 0)*(betas[0] - arma::accu(betas_scaled % mu_x.t()));
}

// Functions to return coefficients and the intercept
arma::vec ProxGrad::Get_Coef(){
  return(betas);
}
arma::vec ProxGrad::Get_Coef_Scaled(){
  return(betas_scaled);
}
double ProxGrad::Get_Intercept(){
  return(betas[0]);
}
double ProxGrad::Get_Intercept_Scaled(){
  return(intercept_scaled);
}

ProxGrad::~ProxGrad(){
  // Class destructor
}

// ----------------------------------
// Static Functions - Gradient Steps
// ----------------------------------

void ProxGrad::AdaGrad(arma::vec & grad_step, arma::vec & grad_vector){
  
  grad_step = arma::ones(grad_vector.n_elem)/arma::norm(grad_vector);
}

/*
* --------------------------------------
* Static Functions - Descent Iterations
* --------------------------------------
*/

// Gradient proposals iterations
void ProxGrad::ISTA(double & t_prev, double & t_next, arma::vec & betas, arma::vec & new_betas, arma::vec & proposal){
  
  t_next = (1 + std::pow(1 + 4*std::pow(t_prev, 2), 0.5))/2;
  proposal = new_betas + ((t_prev - 1)/t_next) * (new_betas - betas);
}

void ProxGrad::FISTA(double & t_prev, double & t_next, arma::vec & betas, arma::vec & new_betas, arma::vec & proposal){
  
  proposal = new_betas;
}

/*
* -----------------------------------------------------------
* Static Functions - (Negative) Log-Likelihoods Computation
* -----------------------------------------------------------
*/

// Static FUnctions - (Negative) Log-Likelihoods Computation
double ProxGrad::Linear_Likelihood(arma::mat & x, arma::vec & y, 
                                   arma::vec & betas){
  
  arma::vec linear_fit = x*betas;
  return(arma::accu(arma::square(linear_fit - y))/(2*y.n_elem));
}

double ProxGrad::Logistic_Likelihood(arma::mat & x, arma::vec & y, 
                                     arma::vec & betas){
  
  arma::vec linear_fit = x*betas;
  return(arma::accu(arma::log(1 + arma::exp(linear_fit)) - linear_fit % y)/y.n_elem);
}

double ProxGrad::Gamma_Likelihood(arma::mat & x, arma::vec & y, 
                                  arma::vec & betas){
  
  arma::vec linear_fit = x*betas;
  return(arma::accu(linear_fit + arma::exp(x*(-betas)) % y)/y.n_elem);
}

double ProxGrad::Poisson_Likelihood(arma::mat & x, arma::vec & y, 
                                    arma::vec & betas){
  
  arma::vec linear_fit = x*betas;
  return(arma::accu(arma::exp(linear_fit) - linear_fit % y)/y.n_elem);
}

/*
* ------------------------------------------
* Static Functions - Gradients Computation
* ------------------------------------------
*/

// Linear GLM - Gradient 
void ProxGrad::Linear_Gradient(arma::mat & x, arma::vec & y, 
                               arma::vec & betas, arma::vec & grad_vector){
  
  grad_vector = x.t()*(x*betas - y)/y.n_elem;
}

// Logistic GLM - Gradient 
void ProxGrad::Logistic_Gradient(arma::mat & x, arma::vec & y, 
                                 arma::vec & betas, arma::vec & grad_vector){
  
  grad_vector = x.t()*(1/(1 + arma::exp(x*(-betas))) - y)/y.n_elem;
}

// Gamma GLM - Gradient 
void ProxGrad::Gamma_Gradient(arma::mat & x, arma::vec & y, 
                              arma::vec & betas, arma::vec & grad_vector){
  
  grad_vector = x.t()*(arma::ones(y.n_elem) - arma::exp(x*(-betas)) % y)/y.n_elem;
}

// Poisson GLM - Gradient 
void ProxGrad::Poisson_Gradient(arma::mat & x, arma::vec & y, 
                                arma::vec & betas, arma::vec & grad_vector){
  
  grad_vector = x.t()*(arma::exp(x*betas) -  y)/y.n_elem;
}


/*
* ------------------------------------------------
* Static Functions - Expected Values Computation
* ------------------------------------------------ 
*/

// Linear GLM - Expected Values 
void ProxGrad::Linear_Expected(arma::mat & x, arma::vec & betas, 
                               arma::vec & expected_val){
  
  expected_val = x*betas;
}

// Logistic GLM - Expected Values 
void ProxGrad::Logistic_Expected(arma::mat & x, arma::vec & betas, 
                                 arma::vec & expected_val){
  
  expected_val = 1/(1 + arma::exp(x*(-betas)));
}

// Gamma GLM - Expected Values 
void ProxGrad::Gamma_Expected(arma::mat & x, arma::vec & betas, 
                              arma::vec & expected_val){
  
  expected_val = arma::exp(x*betas);
}

// Poisson GLM - Expected Values 
void ProxGrad::Poisson_Expected(arma::mat & x, arma::vec & betas, 
                                arma::vec & expected_val){
  
  expected_val = arma::exp(x*betas);
}






