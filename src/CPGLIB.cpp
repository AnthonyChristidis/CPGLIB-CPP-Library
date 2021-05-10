/*
* ===========================================================
* File Type: CPP
* File Name: CPGLIB.cpp
* Package Name: CPGLIB
*
* Created by Anthony-A. Christidis.
* Copyright © Anthony-A. Christidis. All rights reserved.
* ===========================================================
*/

#include "CPGLIB.hpp"
#include "ProxGrad.hpp"

#include <RcppArmadillo.h>

#include "config.h" 

#include <math.h>

// Constants - COMPUTATION 
const static double LS_CONST_PARAMETER = 0.25;
const static double GRID_INTERACTION_MAX_COUNTER = 5;
const static arma::uword DIVERSITY_GRID_SIZE = 50;
const static arma::uword SPARSITY_GRID_SIZE = 50;
static const double MAX_DIVERSITY_TOLERANCE = 1e-5; 

// Constructor for CPGLIB_Diversity
CPGLIB::CPGLIB(arma::mat x, arma::vec y,
               arma::uword & type,
               arma::uword & G,
               arma::uword & include_intercept,
               double alpha_s, double alpha_d,
               double lambda_sparsity, double lambda_diversity,
               arma::uword & permutate_search,
               arma::uword & acceleration,
               double tolerance, arma::uword max_iter):
  x(x), y(y),
  type(type),
  G(G),
  include_intercept(include_intercept),
  alpha_s(alpha_s), alpha_d(alpha_d),
  lambda_sparsity(lambda_sparsity), lambda_diversity(lambda_diversity),
  permutate_search(permutate_search),
  acceleration(acceleration),
  tolerance(tolerance), max_iter(max_iter){
  
  // Initializing the object
  Initialize();
}
void CPGLIB::Initialize(){
  
  // Standardization of design matrix
  mu_x = arma::mean(x);
  sd_x = arma::stddev(x, 1);
  x.each_row() -= mu_x;
  x.each_row() /= sd_x;

  // Setting the parameters
  n = x.n_rows;
  p = x.n_cols + 1;
  betas = arma::zeros(p, G);
  new_betas = arma::zeros(p, G);
  grad_vector = arma::zeros(p);
  expected_val = arma::zeros(n, G);
  active_set = arma::zeros(p);
  new_active_set = arma::zeros(p);
  intercept_scaled = arma::vec(G);
  
  // Augmented matrix for optimization
  if(include_intercept)
    x_std_aug = arma::join_rows(arma::ones(n, 1), x);
  else
    x_std_aug = arma::join_rows(arma::zeros(n, 1), x);
  
  // Setting initial values and function pointers for expected values and weights
  if(type==1){ // Linear Model
    
    Compute_Likelihood = &CPGLIB::Linear_Likelihood;
    Compute_Gradient = &CPGLIB::Linear_Gradient;
    Compute_Expected = &CPGLIB::Linear_Expected;
    
    if(include_intercept)
      betas[0] = arma::mean(y);
  }
  
  else if(type==2){ // Logistic Regression
    
    Compute_Likelihood = &CPGLIB::Logistic_Likelihood;
    Compute_Gradient = &CPGLIB::Logistic_Gradient;
    Compute_Expected = &CPGLIB::Logistic_Expected;
    
    if(include_intercept)
      betas[0] = std::log(arma::mean(y)/(1-arma::mean(y)));
  }
  
  else if(type==3){ // Gamma GLM
    
    Compute_Likelihood = &CPGLIB::Gamma_Likelihood;
    Compute_Gradient = &CPGLIB::Gamma_Gradient;
    Compute_Expected = &CPGLIB::Gamma_Expected; 
    if(include_intercept)
      betas[0] = std::log(arma::mean(y));
  }
  
  else if(type==4){ // Poisson GLM
    
    Compute_Likelihood = &CPGLIB::Poisson_Likelihood;
    Compute_Gradient = &CPGLIB::Poisson_Gradient;
    Compute_Expected = &CPGLIB::Poisson_Expected;
    
    if(include_intercept)
      betas[0] = std::log(arma::mean(y));
  }
  
  // Gradient step sizes options 
  Compute_Gradient_Step = &CPGLIB::AdaGrad;
  
  // Gradient proposals
  if(acceleration)
    Proposal_Iteration = &CPGLIB::FISTA;
  else 
    Proposal_Iteration = &CPGLIB::ISTA;
}

// Iterative Soft function
arma::vec CPGLIB::Soft(arma::vec & u, 
                       arma::vec & threshold, arma::vec & scale){ 
  
  grad_vector = u - threshold;
  grad_vector(arma::find(grad_vector < 0)).zeros();
  return((arma::sign(u) % grad_vector)/(1 + scale));
}

// Function to compute the weights penalty (from diversity penalty) - ABS
arma::vec CPGLIB::Beta_Weights_Abs(arma::uword & group){
  
  arma::vec indices = arma::ones(new_betas.n_cols, 1);
  indices[group] = 0;
  return(arma::abs(new_betas)*indices);
}
// Function to compute the weights penalty (from diversity penalty) - SQ
arma::vec CPGLIB::Beta_Weights_Sq(arma::uword & group){
  
  arma::vec indices = arma::ones(new_betas.n_cols, 1);
  indices[group] = 0;
  return(arma::square(new_betas)*indices);
}

// Function to compute the sparsity penalty
double CPGLIB::Sparsity_Penalty(arma::uword & group){ 
  
  return(lambda_sparsity*((1-alpha_s)*0.5*(arma::accu(arma::pow(betas.col(group), 2)) - std::pow(betas.col(group)[0], 2)) + 
         alpha_s*(arma::accu(arma::abs(betas.col(group))) - std::abs(betas.col(group)[0]))));
}
double CPGLIB::Sparsity_Penalty_New(arma::uword & group){ 
  
  return(lambda_sparsity*((1-alpha_s)*0.5*(arma::accu(arma::pow(new_betas.col(group), 2)) - std::pow(new_betas.col(group)[0], 2)) + 
         alpha_s*(arma::accu(arma::abs(new_betas.col(group))) - std::abs(new_betas.col(group)[0]))));
}

// Function to compute the sparsity (objective function)
double CPGLIB::Sparsity_Penalty(){ 
  
  arma::mat betas_penalty = betas.rows(arma::linspace<arma::uvec>(1, p-1, p-1));
  return(lambda_sparsity*((1-alpha_s)*0.5*pow(arma::norm(betas_penalty, "fro"), 2) + alpha_s*arma::accu(arma::abs(betas_penalty))));
}
// Function to compute the diversity penalty (objective function)
double CPGLIB::Diversity_Penalty(){
  
  arma::mat betas_penalty = betas.rows(arma::linspace<arma::uvec>(1, p-1, p-1));
  double diversity_penalty = 0;
  arma::mat gram_betas = arma::zeros(betas_penalty.n_rows, betas_penalty.n_rows);
  gram_betas = arma::abs(betas_penalty.t())*arma::abs(betas_penalty);
  gram_betas.diag().zeros();
  diversity_penalty = 0.5*arma::accu(gram_betas);
  diversity_penalty *= lambda_diversity;
  
  return(diversity_penalty);
}

// Function to return the active variables
void CPGLIB::Update_Active_Set(){ 
  
  active_set.ones();
  active_set(arma::find(new_betas == 0)).zeros();
  new_active_set.ones();
  new_active_set(arma::find(new_betas == 0)).zeros();
}
bool CPGLIB::Compare_Active_Set(){
  
  return(arma::accu(arma::abs(active_set - new_active_set)) == 0);
}

// Method to set lambda to new value and return current lambda
void CPGLIB::Compute_Lambda_Sparsity_Grid(){
  
  // Grid size parameter
  double eps_sparsity;
  if(n>p)
    eps_sparsity = 1e-4;
  else
    eps_sparsity = 1e-2;
  
  // Standardization of design matrix
  arma::rowvec mu_x = arma::mean(x);
  arma::rowvec sd_x = arma::stddev(x, 1);
  arma::mat x_std = x;
  x_std.each_row() -= mu_x;
  x_std.each_row() /= sd_x;
  
  // Maximum lambda_sparsity that kills all variables
  double lambda_sparsity_max = (1/alpha_s)*arma::max(abs(y.t()*x_std))/n;
  lambda_sparsity_grid =  arma::exp(arma::linspace(std::log(eps_sparsity*lambda_sparsity_max), 
                                                   std::log(lambda_sparsity_max), 
                                                   SPARSITY_GRID_SIZE));
}

// Method to get a diversity penalty parameter that kills all model interactions
void CPGLIB::Compute_Lambda_Diversity_Max(){
  
  // Initial guess for the diversity penalty
  lambda_diversity_max = 3*G;
  
  // Split model to determine the maximum diversity parameter
  CPGLIB betas_grid = CPGLIB(x, y,
                             type,
                             G, 
                             include_intercept,
                             alpha_s, alpha_d, 
                             lambda_sparsity, lambda_diversity_max,
                             permutate_search,
                             acceleration,
                             MAX_DIVERSITY_TOLERANCE, max_iter);
  
  betas_grid.Cycle_Groups_Balanced();
  arma::uword counter = 0;
  // While interactions remain, increase lambda_diversity_max by scaling it by a constant factor of two
  while(Check_Interactions_Beta(betas_grid.Get_Coef_Scaled()) & (counter<=GRID_INTERACTION_MAX_COUNTER)){ 
    
    lambda_diversity_max *= 2;
    betas_grid.Set_Lambda_Diversity(lambda_diversity_max);
    betas_grid.Cycle_Groups_Balanced();
    counter++;
  }

  // Current grid
  double eps_diversity;
  if(n > p)
    eps_diversity = 1e-4;
  else
    eps_diversity = 1e-2; 
  lambda_diversity_grid = arma::exp(arma::linspace(std::log(eps_diversity*lambda_diversity_max), 
                                                   std::log(lambda_diversity_max), 
                                                   DIVERSITY_GRID_SIZE));
  
  // If we could not kill all the interactions
  if(Check_Interactions_Beta(betas_grid.Get_Coef_Scaled())){
    
    Rcpp::warning("Failure to find lambda_diversity that kills all interactions.");
    
  } else{
    
    // Computing the beta coefficients for the candidates in the grid
    arma::cube beta_grid_candidates = arma::zeros(p-1, G, DIVERSITY_GRID_SIZE);
    for(int diversity_ind=DIVERSITY_GRID_SIZE-1; diversity_ind>=0; diversity_ind--){
      
      betas_grid.Set_Lambda_Diversity(lambda_diversity_grid[diversity_ind]);
      betas_grid.Cycle_Groups_Balanced();
      beta_grid_candidates.slice(diversity_ind) = betas_grid.Get_Coef_Scaled();
    }
    
    // Find smallest lambda_diversity in the grid such that there are no interactions
    arma::uvec interactions = Check_Interactions(beta_grid_candidates);
    // Find smallest index where there are no interactions
    arma::uvec indexes = arma::find(interactions==0, 1);
    arma::uword index_max = indexes[0];
    lambda_diversity_max = lambda_diversity_grid[index_max];
  }

}

// Method to get the grid of lambda_diversity
void CPGLIB::Compute_Lambda_Diversity_Grid(){
  
  double eps_diversity;
  if(n > p)
    eps_diversity = 1e-4;
  else
    eps_diversity = 1e-2; 
  
  // Grid computation
  lambda_diversity_grid = arma::exp(arma::linspace(std::log(eps_diversity*lambda_diversity_max), 
                                                   std::log(lambda_diversity_max), 
                                                   DIVERSITY_GRID_SIZE));
  lambda_diversity_grid[0] = 0; // Initialization: no diversity
}

// Function that checks if there are interactions between groups in the matrix of betas
bool CPGLIB::Check_Interactions_Beta(arma::mat betas_matrix){
  
  arma::uword p = betas_matrix.n_rows;
  bool interactions = false;
  for (arma::uword var_ind=0; var_ind<p; var_ind++){
    arma::mat temp = arma::nonzeros(betas_matrix.row(var_ind));
    if(temp.n_rows > 1){
      return(true);
    }
  }
  return(interactions);
}

// Function to returns a vector with ones corresponding to the betas that have interactions.
arma::uvec CPGLIB::Check_Interactions(arma::cube & betas){
  
  arma::vec checks = arma::zeros(betas.n_slices, 1);
  arma::vec all_ones = arma::ones(betas.n_slices, 1);
  for(arma::uword i = 0; i < betas.n_slices; i++){
    checks(i) = Check_Interactions_Beta(betas.slice(i));
  }
  return(checks==all_ones);
}


// Functions to set new data
void CPGLIB::Set_X(arma::mat & x){
  
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
void CPGLIB::Set_Y(arma::vec & y){
  this->y = y;
}

// Functions to set maximum number of iterations and tolerance
void CPGLIB::Set_Max_Iter(arma::uword & max_iter){
  this->max_iter = max_iter;
}
void CPGLIB::Set_Tolerance(double & tolerance){
  this->tolerance = tolerance;
}

// Functions to set and get alpha_s
void CPGLIB::Set_Alpha_Sparsity(double alpha_s){
  this->alpha_s = alpha_s;
}
double CPGLIB::Get_Alpha_Sparsity(){
  return(this->alpha_s);
}
// Method to set alpha_d to new value and return current alpha_d
void CPGLIB::Set_Alpha_Diversity(double alpha_d){
  this->alpha_d = alpha_d;
}
double CPGLIB::Get_Alpha_Diversity(){
  return(this->alpha_d);
}

// Functions to set and get lambda_sparsity
void CPGLIB::Set_Lambda_Sparsity(double lambda_sparsity){
  this->lambda_sparsity = lambda_sparsity;
}
double CPGLIB::Get_Lambda_Sparsity(){
  return(lambda_sparsity);
}
// Functions to set and get lambda_diversity
void CPGLIB::Set_Lambda_Diversity(double lambda_diversity){
  this->lambda_diversity = lambda_diversity;
}
double CPGLIB::Get_Lambda_Diversity(){
  return(this->lambda_diversity);
}

// Function to set coefficients 
void CPGLIB::Set_Betas(arma::uword group, arma::vec set_betas){
  
  this->betas.col(group) = set_betas;
  this->new_betas.col(group) = set_betas;
}

// Function to initialize coefficients - no diversity
void CPGLIB::Initialize_Betas_No_Diversity(){
  
  // ProxGrad model initialization
  ProxGrad single_model = ProxGrad(x, y,
                                   type,
                                   include_intercept,
                                   alpha_s,
                                   lambda_sparsity,
                                   acceleration,
                                   tolerance, max_iter);
  
  // Coefficients Computation
  single_model.Compute_Coef();
  
  // Setting the initial betas
  for(arma::uword group_ind=0; group_ind<G; group_ind++){
    Set_Betas(group_ind, single_model.Get_Coef());
  }
}

// Function to return expected values
arma::vec CPGLIB::Get_Expected(){
  return(expected_val);
}

// Function to compute the objective value
double CPGLIB::Get_Objective_Value(arma::uword & group){
  
  return(Compute_Likelihood(x_std_aug, y, betas, group) + Sparsity_Penalty(group));
}
double CPGLIB::Get_Objective_Value_New(arma::uword & group){
  
  return(Compute_Likelihood(x_std_aug, y, new_betas, group) + Sparsity_Penalty_New(group));
}

// Function to compute the global objective function value
double CPGLIB::Get_Objective_Value(){
  
  // Objective value for all groups
  double objective_val = 0;
  for(arma::uword group_ind=0; group_ind<G; group_ind++)
    objective_val += Get_Objective_Value(group_ind);
  
  // Returning objective value over all groups and the diversity penalty
  return(objective_val + Diversity_Penalty());
}

// Function to compute coefficients
void CPGLIB::Compute_Coef(arma::uword & group){

  // Variables for proximal gradient descent
  arma::vec proposal = betas.col(group);
  iter_count = 0;
  arma::vec prox_vector = arma::ones(p);
  arma::vec prox_threshold, prox_scale;
  arma::vec prox_threshold_step = lambda_sparsity*alpha_s + lambda_diversity*alpha_d*Beta_Weights_Abs(group)/2;
  arma::vec prox_scale_step = lambda_sparsity*(1-alpha_s) + lambda_diversity*(1-alpha_d)*Beta_Weights_Sq(group)/2;
  
  // Variables for Nesterov acceleration
  double t_prev = 1, t_next = 1;
  
  // Initial gradient computation
  Compute_Gradient(x_std_aug, y, proposal, grad_vector);
  Compute_Gradient_Step(grad_step, grad_vector);
  
  // Proximal gradient descent iterations
  do{
    
    betas.col(group) = new_betas.col(group);
    Compute_Gradient(x_std_aug, y, proposal, grad_vector);
    Compute_Gradient_Step(grad_step, grad_vector);
    
    // Line search for (local) descent (with acceleration)
    do{
      
      // Proximal gradient descent starma::mat & new_betasep
      prox_vector = proposal - grad_step % grad_vector;
      prox_threshold = grad_step % prox_threshold_step;
      prox_scale = grad_step % prox_scale_step;
      new_betas.col(group) = Soft(prox_vector, prox_threshold, prox_scale);
      new_betas.col(group)[0] = prox_vector[0]; // Adjustment for intercept term (no shrinkage)
      
      // Line search adjustment
      grad_step = grad_step*LS_CONST_PARAMETER;
      
    } while (Get_Objective_Value_New(group) > Get_Objective_Value(group));
    
    // (!) The above conditional can be replaced by the quadratic approximation 
    
    // Iteration proposal
    Proposal_Iteration(t_prev, t_next,  
                       betas, new_betas, proposal,
                       group); 
    
    // Iteration for algorithm
    iter_count++;

  } while ((iter_count <= max_iter) && (arma::max(arma::square(new_betas.col(group) - betas.col(group))) > tolerance));
  
  // Final coefficients
  betas.col(group) = new_betas.col(group);
}

// Cycling over the groups
void CPGLIB::Cycle_Groups(){
  
  for(arma::uword group_ind=0; group_ind<G; group_ind++)
    Compute_Coef(group_ind);
  
  // Permutation search
  if(permutate_search){
    
    // Variable to store indicator
    arma::uword permutate_iter_cycles = std::ceil(G/2.0);
    arma::uword group_id;
    double objective_old;
    arma::mat betas_old;
    
    for(arma::uword permutate_iter=0; permutate_iter<permutate_iter_cycles; permutate_iter++){
      
      // Setting old betas
      objective_old = Get_Objective_Value();
      betas_old = betas;
      
      // Generating sampling order
      Rcpp::IntegerVector group_order = Rcpp::sample(G, G, false);
      
      for(arma::uword group_ind=0; group_ind<G; group_ind++){
        group_id = group_order[group_ind]-1;
        Compute_Coef(group_id); 
      }

      // Conditional cycling update
      if(Get_Objective_Value()>objective_old)
        betas = new_betas = betas_old;
    }
  } // End of permutation cycling
  
  // Scaling coefficients
  Scale_Coefficients();
  Scale_Intercept();
}

// Cycling function for non-CV application
void CPGLIB::Cycle_Groups_Grid(){
  
  // No diversity case
  if(lambda_diversity==0){
    
    // Initialization of the coefficients
    Initialize_Betas_No_Diversity();
    // Scaling coefficients
    Scale_Coefficients();
    Scale_Intercept();
    return;
  }

  // Computing the grid for the diversity parameter
  Compute_Lambda_Diversity_Max();
  Compute_Lambda_Diversity_Grid();
  
  // Setting the set lambda_diversity
  double lambda_diversity_set = lambda_diversity;

  // Cycling over diversity parameter (stability)
  for(arma::uword diversity_ind=1; diversity_ind<lambda_diversity_grid.n_elem; diversity_ind++){
    
    Set_Lambda_Diversity(lambda_diversity_grid[diversity_ind]);
    if(lambda_diversity>=lambda_diversity_set)
      break;
    Cycle_Groups();
  }
    
  // Final fit
  Set_Lambda_Diversity(lambda_diversity_set);
  Cycle_Groups();
}

// Function to update coefficients (balanced cycling)
void CPGLIB::Cycle_Groups_Balanced(){
  
  // Variables for proximal gradient descent
  arma::vec proposal = arma::zeros(p);
  arma::vec prox_vector = arma::ones(p);
  arma::vec prox_threshold = arma::zeros(p), prox_scale = arma::zeros(p);
  arma::vec prox_threshold_step = arma::zeros(p), prox_scale_step = arma::zeros(p);
  
  // Iterations for cycling
  for(arma::uword iter_cycling = 0; iter_cycling<max_iter; iter_cycling++){
  
    // Cycling over the groups
    for(arma::uword group = 0; group<G; group++){
    
      // Variables for proximal gradient descent
      proposal = betas.col(group);
      prox_threshold_step = lambda_sparsity*alpha_s + lambda_diversity*alpha_d*Beta_Weights_Abs(group)/2;
      prox_scale_step = lambda_sparsity*(1-alpha_s) + lambda_diversity*(1-alpha_d)*Beta_Weights_Sq(group)/2;
      
      // Gradient computation
      Compute_Gradient(x_std_aug, y, proposal, grad_vector);
      Compute_Gradient_Step(grad_step, grad_vector);

      // Line search for (local) descent (with acceleration)
      do{
        
        // Proximal gradient descent starma::mat & new_betasep
        prox_vector = proposal - grad_step % grad_vector;
        prox_threshold = grad_step % prox_threshold_step;
        prox_scale = grad_step % prox_scale_step;
        new_betas.col(group) = Soft(prox_vector, prox_threshold, prox_scale);
        new_betas.col(group)[0] = prox_vector[0]; // Adjustment for intercept term (no shrinkage)

        // Line search adjustment
        grad_step = grad_step*LS_CONST_PARAMETER;
        
      } while (Get_Objective_Value_New(group) > Get_Objective_Value(group));
      
    } // End of cycling over the groups
    
    // End of proximal gradient descent if variables are already converged
    if(arma::square(arma::mean(new_betas,1)-arma::mean(betas,1)).max()<tolerance){
      
      iter_count = iter_cycling;
      betas = new_betas;
      Scale_Coefficients();
      Scale_Intercept();
      return;
    }

    // Iteration for cycling count
    iter_count++;
    betas = new_betas;

  } // End of cycling iterations
  
  // Scaling of coefficients
  Scale_Coefficients();
  Scale_Intercept();
  
}

// Cycling function for non-CV application
void CPGLIB::Cycle_Groups_Balanced_Grid(){
  
  // No diversity case
  if(lambda_diversity==0){
    
    // Initialization of the coefficients
    Initialize_Betas_No_Diversity();
    // Scaling coefficients
    Scale_Coefficients();
    Scale_Intercept();
    return;
  }
  
  // // Computing the grid for the diversity parameter
  // Compute_Lambda_Diversity_Max();
  // Compute_Lambda_Diversity_Grid();
  
  // Compute the grid over the sparsity parameter
  Compute_Lambda_Sparsity_Grid();
  
  // Setting the set lambda_diversity
  double lambda_sparsity_set = lambda_sparsity;
  
  // Cycling over diversity parameter (stability) 
  for(arma::uword sparsity_ind=lambda_sparsity_grid.n_elem-1; sparsity_ind>0; sparsity_ind--){
     
    Set_Lambda_Sparsity(lambda_sparsity_grid[sparsity_ind]);
    if(lambda_sparsity<=lambda_sparsity_set)
      break;
    Cycle_Groups_Balanced();
  }
  
  // Final fit
  Set_Lambda_Sparsity(lambda_sparsity_set);
  Cycle_Groups_Balanced();
}

// Function to return the number of iterations for convergence
arma::uword CPGLIB::Get_Iter(){
  return(iter_count);
}

// Function to scale back coefficients to original scale
void CPGLIB::Scale_Coefficients(){
  
  betas_scaled = betas.rows(arma::linspace<arma::uvec>(1, p-1, p-1));
  betas_scaled.each_col() /= (sd_x.t());
}
void CPGLIB::Scale_Intercept(){
  
  for(arma::uword group_ind=0; group_ind<G; group_ind++)
    intercept_scaled[group_ind] = ((include_intercept) ? 1 : 0)*(betas.col(group_ind)[0] - arma::accu(betas_scaled.col(group_ind) % mu_x.t()));
}

// Functions to return coefficients and the intercept
arma::mat CPGLIB::Get_Coef(){
  return(betas);
}
arma::mat CPGLIB::Get_Coef_Scaled(){
  return(betas_scaled);
}
arma::rowvec CPGLIB::Get_Intercept(){
  return(betas.row(0));
}
arma::vec CPGLIB::Get_Intercept_Scaled(){
  return(intercept_scaled); 
}

CPGLIB::~CPGLIB(){
  // Class destructor
}

// ----------------------------------
// Static Functions - Gradient Steps_diversity
// ----------------------------------

void CPGLIB::AdaGrad(arma::vec & grad_step, arma::vec & grad_vector){
  
  grad_step = arma::ones(grad_vector.n_elem)/arma::norm(grad_vector);
}

/*
* --------------------------------------
* Static Functions - Descent Iterations
* --------------------------------------
*/

// Gradient proposals iterations
void CPGLIB::ISTA(double & t_prev, double & t_next, 
                  arma::mat & betas, arma::mat & new_betas, arma::vec & proposal,
                  arma::uword & group){
  
  t_next = (1 + std::pow(1 + 4*std::pow(t_prev, 2), 0.5))/2;
  proposal = new_betas.col(group) + ((t_prev - 1)/t_next) * (new_betas.col(group) - betas.col(group));
}

void CPGLIB::FISTA(double & t_prev, double & t_next, 
                   arma::mat & betas, arma::mat & new_betas, arma::vec & proposal,
                   arma::uword & group){
  
  proposal = new_betas.col(group);
}

/*
* -----------------------------------------------------------
* Static Functions - (Negative) Log-Likelihoods Computation
* -----------------------------------------------------------
*/

// Static FUnctions - (Negative) Log-Likelihoods Computation
double CPGLIB::Linear_Likelihood(arma::mat & x, arma::vec & y, 
                                 arma::mat & betas, 
                                 arma::uword & group){
  
  arma::vec linear_fit = x*betas.col(group);
  return(arma::accu(arma::square(linear_fit - y))/(2*y.n_elem));
}

double CPGLIB::Logistic_Likelihood(arma::mat & x, arma::vec & y, 
                                   arma::mat & betas, 
                                   arma::uword & group){
  
  arma::vec linear_fit = x*betas.col(group);
  return(arma::accu(arma::log(1 + arma::exp(linear_fit)) - linear_fit % y)/y.n_elem);
}

double CPGLIB::Gamma_Likelihood(arma::mat & x, arma::vec & y, 
                                arma::mat & betas, 
                                arma::uword & group){
  
  arma::vec linear_fit = x*betas.col(group);
  return(arma::accu(linear_fit + arma::exp(x*(-betas)) % y)/y.n_elem);
}

double CPGLIB::Poisson_Likelihood(arma::mat & x, arma::vec & y, 
                                  arma::mat & betas, 
                                  arma::uword & group){
  
  arma::vec linear_fit = x*betas.col(group);
  return(arma::accu(arma::exp(linear_fit) - linear_fit % y)/y.n_elem);
}

/*
* ------------------------------------------
* Static Functions - Gradients Computation
* ------------------------------------------
*/

// Linear GLM - Gradient 
void CPGLIB::Linear_Gradient(arma::mat & x, arma::vec & y, 
                             arma::mat & betas, arma::vec & grad_vector){ 
  
  arma::vec linear_fit = x*betas;
  grad_vector = x.t()*(linear_fit - y)/y.n_elem;
}

// Logistic GLM - Gradient 
void CPGLIB::Logistic_Gradient(arma::mat & x, arma::vec & y, 
                               arma::mat & betas, arma::vec & grad_vector){
  
  arma::vec linear_fit = x*betas;
  grad_vector = x.t()*(1/(1 + arma::exp(-linear_fit)) - y)/y.n_elem;
}

// Gamma GLM - Gradient 
void CPGLIB::Gamma_Gradient(arma::mat & x, arma::vec & y, 
                            arma::mat & betas, arma::vec & grad_vector){
  
  arma::vec linear_fit = x*betas;
  grad_vector = x.t()*(arma::ones(y.n_elem) - arma::exp(-linear_fit) % y)/y.n_elem;
}

// Poisson GLM - Gradient 
void CPGLIB::Poisson_Gradient(arma::mat & x, arma::vec & y, 
                              arma::mat & betas, arma::vec & grad_vector){
  
  arma::vec linear_fit = x*betas;
  grad_vector = x.t()*(arma::exp(linear_fit) -  y)/y.n_elem;
}


/*
 * ------------------------------------------------
 * Static Functions - Expected Values Computation
 * ------------------------------------------------ 
 */

// Linear GLM - Expected Values 
void CPGLIB::Linear_Expected(arma::mat & x, arma::mat & betas, 
                             arma::vec & expected_val,
                             arma::uword & group){
  
  expected_val.col(group) = x*betas.col(group);
}

// Logistic GLM - Expected Values 
void CPGLIB::Logistic_Expected(arma::mat & x, arma::mat & betas, 
                               arma::vec & expected_val,
                               arma::uword & group){
  
  expected_val.col(group) = 1/(1 + arma::exp(x*(-betas.col(group))));
}

// Gamma GLM - Expected Values 
void CPGLIB::Gamma_Expected(arma::mat & x, arma::mat & betas, 
                            arma::vec & expected_val,
                            arma::uword & group){
  
  expected_val.col(group) = arma::exp(x*betas.col(group));
}

// Poisson GLM - Expected Values 
void CPGLIB::Poisson_Expected(arma::mat & x, arma::mat & betas, 
                              arma::vec & expected_val,
                              arma::uword & group){
  
  expected_val.col(group) = arma::exp(x*betas.col(group));
}






