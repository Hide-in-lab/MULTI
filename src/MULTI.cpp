#include <RcppArmadillo.h>
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// Adaptive lamda
double compute_lambda( const arma::mat& Sigma, double min_lambda = 1e-4 ) {
  arma::vec eigval;
  arma::eig_sym( eigval, Sigma );
  double cond_number = eigval.max() / std::max( eigval.min(), 1e-16 );
  return std::max( 1.0 / cond_number, min_lambda );
}

// Calculate first-order partial derivative with respect to sigma2_theta
double der_sigma2_theta(
    double sigma2_theta, 
    const  arma::vec& mu_theta,     const arma::mat& Sigma2_theta,
    const  arma::vec& eigval_SGRSG, const arma::mat& eigvec_SGRSG ){
  int K = mu_theta.n_elem;
  arma::vec adjusted_eigval = 1.0/( eigval_SGRSG + sigma2_theta );
  arma::vec temp = eigvec_SGRSG.t() * mu_theta;
  arma::mat Sigma_transformed = eigvec_SGRSG.t() * Sigma2_theta * eigvec_SGRSG;
  
  double term1 = arma::sum( square( temp ) % square( adjusted_eigval ) );
  double term2 = 0.0;
  for( int i = 0; i < K; ++i ){
    term2 += Sigma_transformed( i, i ) * adjusted_eigval( i ) * adjusted_eigval( i );
  }
  double term3 = arma::sum( adjusted_eigval );

  return ( term1 + term2 - term3 )/2;
}

// Solve the root of sigma2_theta
double find_sigma2_theta(
    const  arma::vec& mu_theta,     const arma::mat& Sigma2_theta, 
    const  arma::vec& eigval_SGRSG, const arma::mat& eigvec_SGRSG,
    double low = -1e-6, double high = 1e2, double tol = 1e-4, int max_iter = 1000 ) {
  
  double flow, fhigh;
  flow  = der_sigma2_theta( low,  mu_theta, Sigma2_theta, eigval_SGRSG, eigvec_SGRSG);
  fhigh = der_sigma2_theta( high, mu_theta, Sigma2_theta, eigval_SGRSG, eigvec_SGRSG);

  double mid;
  for( int i = 0; i < max_iter; ++i ){
    mid = ( low + high )/2.0;
    double fmid;
    fmid = der_sigma2_theta( mid, mu_theta, Sigma2_theta, eigval_SGRSG, eigvec_SGRSG);
  
    if( std::abs( fmid ) < tol || ( high - low ) < tol ) return mid;
    if( flow * fmid < 0) {
      high = mid; fhigh = fmid;
    } else {
      low = mid; flow = fmid;
    }
  }

  return mid;
}

// Calculate ELBO
double ELBO_cal( 
    arma::mat A_gamma, arma::rowvec B_gamma, arma::vec mu_gamma, arma::mat Sigma2_gamma,
    arma::mat A_Gamma, arma::rowvec B_Gamma, arma::vec mu_Gamma, arma::mat Sigma2_Gamma,
    arma::vec mu_kappay, arma::mat Sigma2_kappay,
    double mu_beta,      double Sigma2_beta,
    double sigma2_gamma, double sigma2_kappay, double sigma2_beta ){
  arma::vec l( 1, fill::zeros ); int K = mu_kappay.size();
  //---- +E_q_[ log P( gammahat, S_gamma, R_hat | gamma ) ] ----//
  l =   - 0.5 * ( - 2 * B_gamma * mu_gamma + mu_gamma.t() * A_gamma * mu_gamma + arma::trace( A_gamma * Sigma2_gamma ) );
  //-------------------------------------------//
  //---- +E_q_[ log P( Gammahat, S_Gamma, R_hat | gamma, kappay ) ] ----//
  l = l - 0.5 * ( - 2 * B_Gamma * mu_Gamma + mu_Gamma.t() * A_Gamma * mu_Gamma + arma::trace( A_Gamma * Sigma2_Gamma ) );
  //---- +E_q_[ log( gamma | sigma2_gamma )   ] ----//
  l = l - arma::sum( arma::square( mu_gamma ) + Sigma2_gamma.diag( )  )  / ( 2 * sigma2_gamma )  - K/2 * std::log( sigma2_gamma );
  //---- +E_q_[ log( kappay | sigma2_kappay )   ] ----//
  l = l - arma::sum( arma::square( mu_kappay ) + Sigma2_kappay.diag( ) ) / ( 2 * sigma2_kappay ) - K/2 * std::log( sigma2_kappay );
  //---- + E_q_[ log( beta | sigma2_beta )   ] ----//
  l = l - ( mu_beta * mu_beta + Sigma2_beta ) / ( 2 * sigma2_beta ) - 1/2 * std::log( sigma2_beta );
  //---- -E( log q( gamma ) ) ----//
  l = l + 0.5 * arma::sum( log( Sigma2_gamma.diag( ) ) );
  //---- -E( log q( kappay ) ) ----//
  l = l + 0.5 * arma::sum( log( Sigma2_kappay.diag( ) ) );
  //---- -E( log q( betq ) ) ----//
  l = l + 0.5 * log( Sigma2_beta );
  //-------------------------------------------//
  double output;
  return output = arma::as_scalar( l );
}

// Calculate Hellinger Distance
double hellinger_distance( double mu1, double se1, double mu2, double se2 ){
  double var1 = se1 * se1; double var2 = se2 * se2;
  double term1 = ( 2.0 * se1 * se2 )/( var1 + var2 );
  double term2 = exp( - pow( mu1 - mu2, 2 )/( 4.0 * ( var1 + var2 ) ) );
  double H_sq = 1.0 - sqrt(term1) * term2;
  return sqrt( H_sq );
}


// [[Rcpp::export]]
Rcpp::List lm_cpp(const Eigen::Map<Eigen::VectorXd> y, const Eigen::Map<Eigen::MatrixXd> X) {
  int n = y.size();
  int times = X.cols();
  
  Eigen::VectorXd coef(times);
  Eigen::VectorXd std(times);
  
  Eigen::MatrixXd matrix(n, 2);
  matrix.col(0).setOnes();
  
  for (int i = 0; i < times; ++i) {
    matrix.col(1) = X.col(i); 

    Eigen::VectorXd coef_vec = matrix.colPivHouseholderQr().solve(y);

    Eigen::VectorXd res = y - matrix * coef_vec;
    double rss = res.squaredNorm();
    double std_error = std::sqrt(rss / (n - 2));

    Eigen::VectorXd se_coef = std_error * (matrix.transpose() * matrix).inverse().diagonal().array().sqrt();
    
    coef[i] = coef_vec(1); 
    std[i] = se_coef(1);  
  }
  
  Rcpp::NumericMatrix coef_matrix(times, 1);
  Rcpp::NumericMatrix std_matrix(times, 1);
  for (int i = 0; i < times; ++i) {
    coef_matrix(i, 0) = coef[i];
    std_matrix(i, 0) = std[i];
  } 
  
  return Rcpp::List::create(
    Rcpp::Named("coef") = coef_matrix,
    Rcpp::Named("std") = std_matrix
  );
} 


// [[Rcpp::export]]
List MULTI_single( const List& DataList, 
                   double r2 = 0.01, 
                   int iter_times = 500, 
                   double ELBO_tol = 1e-6 ) {
  /////----------------- Data Preparation -----------------/////
  arma::vec gammahat = DataList[ "bx" ];
  arma::vec Gammahat = DataList[ "by" ];
  arma::mat S_gamma  = diagmat(  as<arma::mat>( DataList[ "bxse" ] ) );
  arma::mat S_Gamma  = diagmat(  as<arma::mat>( DataList[ "byse" ] ) );
  arma::mat R_hat    = as<arma::mat>( DataList[ "R_hat" ] );
  int       K        = gammahat.size();
  /////----------------- Preprocessing Correlation Matrix -----------------/////
  R_hat.elem( find( abs( R_hat ) < sqrt( r2 ) ) ).zeros();
  /////----------------- Precalculate the constant values -----------------/////
  arma::mat Sgi = diagmat( 1.0 / as<arma::mat>( DataList[ "bxse" ] ) ); 
  arma::mat SGi = diagmat( 1.0 / as<arma::mat>( DataList[ "byse" ] ) );
  double lambda = 1e-4;
  arma::mat eyeK = arma::eye( K, K );
  arma::mat SgRSg  = S_gamma * R_hat * S_gamma + lambda * eyeK;  // Regularization
  // arma::mat SgRSg  = S_gamma * R_hat * S_gamma;
  arma::mat SgRSgi = S_gamma * R_hat * Sgi;
  arma::vec eigval_SgRSg;
  arma::mat eigvec_SgRSg;
  arma::eig_sym( eigval_SgRSg, eigvec_SgRSg, SgRSg );
  // eigval_SgRSg = arma::clamp( eigval_SgRSg, 1e-4, arma::datum::inf );
  arma::mat iSgRSg = eigvec_SgRSg * diagmat( 1.0/eigval_SgRSg ) * eigvec_SgRSg.t();
  /////--------------------------------------------------------------------/////
  arma::mat SGRSG  = S_Gamma * R_hat * S_Gamma + lambda * eyeK;  // Regularization
  // arma::mat SGRSG  = S_Gamma * R_hat * S_Gamma;
  arma::mat SGRSGi = S_Gamma * R_hat * SGi;
  arma::vec eigval_SGRSG;
  arma::mat eigvec_SGRSG;
  // eigval_SGRSG = arma::clamp( eigval_SGRSG, 1e-4, arma::datum::inf );
  arma::eig_sym( eigval_SGRSG, eigvec_SGRSG, SGRSG );
  /////--------------------------------------------------------------------/////
  arma::mat A_gamma = Sgi          * R_hat * Sgi; // = SgRSgi.t()   * iSgRSg * SgRSgi;
  arma::mat B_gamma = gammahat.t() * Sgi   * Sgi; // = gammahat.t() * iSgRSg * SgRSgi;
  // arma::mat    SgiRSgi = Sgi          * R_hat * Sgi; // = A_gamma
  // arma::mat    A_gamma = SgRSgi.t()   * iSgRSg * SgRSgi;
  // arma::rowvec gSgiSgi = gammahat.t() * Sgi   * Sgi; // = B_gamma
  // arma::rowvec B_gamma = gammahat.t() * iSgRSg * SgRSgi;
  /////--------------------- Parameters Initializing ---------------------/////
  double mu_beta = 0, Sigma2_beta = 1, sigma2_beta = 1;
  arma::vec mu_gamma( K, fill::zeros ), 
  mu_kappay( K, fill::zeros ),  
  mu_theta(  K, fill::zeros ),
  mu_Gamma(  K, fill::zeros );
  arma::mat Sigma2_gamma( K, K, fill::eye ),
  Sigma2_kappay( K, K, fill::eye ),
  Sigma2_theta(  K, K, fill::eye ),
  Sigma2_Gamma(  K, K, fill::eye ); 
  double sigma2_gamma = 1, sigma2_kappay = 1, sigma2_theta = 1;
  std::vector<double> ELBO_set;
  std::vector<double> mu_beta_trace;

  for( int iter = 0; iter < iter_times; ++ iter ){
    /////--------------------- Adaptive regularization update ---------------------/////
    // if( reg_adp && iter%50 == 0 ){
    //   lambda = compute_lambda( Sigma2_gamma );
    //   SgRSg = S_gamma * R_hat * S_gamma + lambda * eyeK;
    //   SGRSG = S_Gamma * R_hat * S_Gamma + lambda * eyeK;
    //   arma::eig_sym( eigval_SgRSg, eigvec_SgRSg, SgRSg );
    //   arma::eig_sym( eigval_SGRSG, eigvec_SGRSG, SGRSG );
    // }
    /////------------------------- E-step -------------------------/////
    //---------------------------------------------------------------//
    arma::mat iSGRSGt = eigvec_SGRSG * diagmat( 1.0/( eigval_SGRSG + sigma2_theta ) ) * eigvec_SGRSG.t();
    arma::mat    A = SGRSGi.t()   * iSGRSGt * SGRSGi;  // = A_Gamma;
    arma::rowvec B = Gammahat.t() * iSGRSGt * SGRSGi;  // = B_Gamma;
    //---------------------------------------------------------------//
    //---------------------------------------------------------------//
    //----- beta -----//
    // Sigma2_beta = 1.0/( 1.0/sigma2_beta + arma::trace( A * Sigma2_gamma ) + arma::dot( mu_gamma, A * mu_gamma ) );
    Sigma2_beta = 1.0/( 1.0/sigma2_beta + arma::accu( A % Sigma2_gamma ) + arma::dot( mu_gamma, A * mu_gamma ) );
    mu_beta     = ( arma::dot( B, mu_gamma ) - arma::dot( mu_gamma, A * mu_kappay ) ) * Sigma2_beta;
    //---------------//
    //----- gamma -----//
    arma::mat    AE_gamma =  A_gamma + ( mu_beta * mu_beta + Sigma2_beta ) * A + eyeK/sigma2_gamma;
    arma::rowvec BE_gamma =  B_gamma + mu_beta * B - mu_beta * mu_kappay.t( ) * A;
    // Sigma2_gamma = pinv( AE_gamma );
    // mu_gamma     = ( BE_gamma * Sigma2_gamma ).t( );
    arma::mat R_gamma = arma::chol( AE_gamma );
    mu_gamma     = arma::solve( R_gamma, solve( R_gamma.t(), BE_gamma.t() ) );
    Sigma2_gamma = arma::solve( R_gamma, solve( R_gamma.t(), eyeK ) );
    //-----------------//
    //----- kappay ----//
    arma::mat    AE_kappay = A + eyeK / sigma2_kappay;
    arma::rowvec BE_kappay = B - mu_beta * mu_gamma.t( ) * A;
    // Sigma2_kappay = pinv( AE_kappay );
    // mu_kappay     = ( BE_kappay * Sigma2_kappay ).t( );
    arma::mat R_kappay = arma::chol( AE_kappay );
    mu_kappay     = arma::solve( R_kappay, solve( R_kappay.t(), BE_kappay.t() ) );
    Sigma2_kappay = arma::solve( R_kappay, solve( R_kappay.t(), eyeK ) );
    //-----------------//
    //-------------------------------------------//

    /////--------------- M-step ---------------/////
    //-------------------------------------------//
    //----- theta -----//
    mu_Gamma     = mu_beta * mu_gamma + mu_kappay;
    mu_theta     = Gammahat - SGRSGi * mu_Gamma;
    Sigma2_Gamma = Sigma2_beta * ( Sigma2_gamma + mu_gamma * mu_gamma.t( ) ) + mu_beta * mu_beta * Sigma2_gamma + Sigma2_kappay;
    Sigma2_theta = SGRSGi * Sigma2_Gamma * SGRSGi.t( );
    // mu_theta = Gammahat - SGRSGi * ( mu_beta * mu_gamma + mu_kappay );
    // Sigma2_theta = SGRSGi * (Sigma2_beta * ( Sigma2_gamma + mu_gamma * mu_gamma.t( ) ) + mu_beta * mu_beta * Sigma2_gamma + Sigma2_kappay) * SGRSGi.t( );
    sigma2_theta = find_sigma2_theta( mu_theta, Sigma2_theta, eigval_SGRSG, eigvec_SGRSG );
    //-----------------//
    sigma2_beta   = mu_beta * mu_beta + Sigma2_beta;
    sigma2_gamma  = arma::as_scalar( ( arma::dot( mu_gamma,  mu_gamma  ) + arma::trace( Sigma2_gamma  ) ) / K );
    sigma2_kappay = arma::as_scalar( ( arma::dot( mu_kappay, mu_kappay ) + arma::trace( Sigma2_kappay ) ) / K );
    // sigma2_beta   = std::max( mu_beta*mu_beta + Sigma2_beta, 1e-4 );
    // sigma2_gamma  = std::max( arma::as_scalar( ( arma::dot( mu_gamma,  mu_gamma  ) + arma::trace( Sigma2_gamma  ) ) / K ), 1e-4 );
    // sigma2_kappay = std::max( arma::as_scalar( ( arma::dot( mu_kappay, mu_kappay ) + arma::trace( Sigma2_kappay ) ) / K ), 1e-4 );
    //-------------------------------------------//
    arma::mat    A_Gamma = SGRSGi.t()   * iSGRSGt * SGRSGi;
    arma::rowvec B_Gamma = Gammahat.t() * iSGRSGt * SGRSGi;
    mu_beta_trace.push_back( mu_beta );
    ELBO_set.push_back( ELBO_cal(
        A_gamma, B_gamma, mu_gamma, Sigma2_gamma,
        A_Gamma, B_Gamma, mu_Gamma, Sigma2_Gamma,
        mu_kappay, Sigma2_kappay,
        mu_beta,  Sigma2_beta, sigma2_gamma, sigma2_kappay, sigma2_beta ) );
   if( iter > 0 && std::abs( ( ELBO_set[ iter ] - ELBO_set[ iter - 1] ) / ELBO_set[ iter - 1 ] ) < ELBO_tol ) break;

  }

  return List::create(
    Named( "mu_beta" )  = mu_beta,
    Named( "se_beta" )  = sqrt( Sigma2_beta + 1e-5 )  // sqrt( Sigma2_beta + 1e-6 )
    // Named( "beta_trace" ) = mu_beta_trace,
    // Named( "ELBO_set" ) = ELBO_set
  );

}

// [[Rcpp::export]]
List MULTI( const List& DataList, std::string tissue_ref = "Tissue1", double r2 = 0.01,
            double cut_off = 0.5, int iter_times = 500, double ELBO_tol = 1e-6 ) {

  int num_data = DataList.size();
  List output( num_data + 2 );
  CharacterVector output_names( num_data + 2 );

  for (int i = 0; i < num_data; ++i) {
    Rcpp::List res = MULTI_single( DataList[i], r2 , iter_times, ELBO_tol );
    output[i] = Rcpp::List::create(
      Rcpp::Named("mu_beta") = res["mu_beta"],
      Rcpp::Named("se_beta") = res["se_beta"]
    );
    output_names[i] = "Estimands for Tissue " + std::to_string(i + 1);
  }

  Rcpp::NumericMatrix HD( num_data, num_data );
  CharacterVector dim_names( num_data );
  for( int i = 0; i < num_data; ++i){
    dim_names[ i ] = "Tissue" + std::to_string( i+ 1 );
    List ref = output[i]; 
    double mu_ref = as<double>(ref["mu_beta"]);
    double se_ref = as<double>(ref["se_beta"]);
    for (int j = i + 1; j < num_data; ++j) {
      List cur = output[j];
      double mu_cur = as<double>(cur["mu_beta"]);
      double se_cur = as<double>(cur["se_beta"]);
      double dist = hellinger_distance( mu_ref, se_ref, mu_cur, se_cur );
      HD(i, j) = dist;
      HD(j, i) = dist;
    }
  }
  HD.attr( "dimnames" ) = Rcpp::List::create( dim_names, dim_names );

  output[ num_data ] = HD;
  output_names[ num_data ] = "Hellinger Distance Matrix";

  int ref_index = std::stoi( tissue_ref.substr( 6 ) ) - 1;
  CharacterVector selected;
  for( int j = 0; j < num_data; ++ j ){
    if( HD( ref_index, j ) < cut_off ) selected.push_back( "Tissue" + std::to_string( j + 1 ) );
  }
  output[ num_data + 1 ] = selected;
  output_names[num_data + 1 ] = "Recommended tissue(s) under Hellinger distance cutoff=" + std::to_string(cut_off) + " (" + tissue_ref + " as reference)";

  output.attr("names") = output_names;
  return output;
} 
