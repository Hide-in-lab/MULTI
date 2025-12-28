run_task <- function( k, h2_kappa, h2_gamma ){    ### Check 1
  
  #####----- Load R packages -----#####
  # library( MULTI )
  # library( MASS )
  # library( MendelianRandomization )
  # library( MCMCpack )
  # library( dlm )
  # library( distr )
  # library( progress )
  # library( dplyr )
  # library( parallel )
  library( callr )
  # Rcpp::sourceCpp( 'MULTI.cpp' )
  
  rp <- r_bg(
    
    func = function( k, h2_kappa, h2_gamma ) {   ### Check 2
      
      library( MASS )
      library( MendelianRandomization )
      library( MCMCpack )
      library( dlm )
      library( distr )
      library( progress )
      library( dplyr )
      library( parallel )
      library( callr )
      Rcpp::sourceCpp( 'MULTI.cpp' )
      
      simulate_once <- function( seed, h2_kappa, h2_gamma ) {   ### Check 3
        set.seed( seed )
        
        #####----- Main Function -----#####
        
        h2_delta  <- h2_gamma*4/5; h2_kappax <- h2_gamma/5
        beta_true <- ifelse( h2_gamma == 0, 0, 0.01 )
        
        #####----- Customed functions -----#####
        #---- 1. Generate Continuous Genotype Matrix -----#
        genotype_continuous <- function( sample_size, nsnp, r_LD ){
          # sample_size <- 10000; nsnp <- 10; r_LD <- 0.2
          miu <- rep(0, nsnp)   ### Set mean variable
          Sigma <- matrix( 1:nsnp^2, ncol = nsnp )   #### Set initial covariance matrix
          for( i in 1:nsnp ){
            Sigma[i, ] <- r_LD^abs( i - (1:nsnp) )   ### Calculate the covariance matrix
          }  ### Elements for the i-th row are r^| i - n |
          return( mvrnorm( sample_size, miu, Sigma ) )
        }
        #---- 2. Convert Continuous Matrix to Factoral Matrix -----#
        genotype_to_factor  <- function( gene_matrix, minor_allel_frequency = c() ){
          MAF_cut <- function( x ){
            MAF <- runif( 1, minor_allel_frequency[1], minor_allel_frequency[2] )
            qt_aa <- MAF^2
            qt_Aa <- MAF*(2 - MAF)   ## 0___aa(MAF*MAF)___Aa(MAF*(2-MAF))___1 ##
            as.integer( cut(x, breaks = c(-Inf, quantile( x, c(qt_aa, qt_Aa) ), Inf), labels = c(0, 1, 2) ) ) - 1
          }
          G <- apply(gene_matrix, 2, MAF_cut)
          return(G)
        }
        #---- 3. Data Generating Function -----#
        data_generating <- function(
    nGTEx = 10000, nGWAS = 10000,
    nsnp = c( 100, 100 ), nconf = 10,  ## nconf for number of confounders ##
    r_kappa = c( 0.2, 0.2 ),
    r_LD = c( 0.5, 0.5 ),
    h2 = list(
      h2_delta1  = h2_delta,  h2_delta2  = h2_delta,
      h2_kappax1 = h2_kappax, h2_kappax2 = h2_kappax,
      h2_kappay1 = 0.05, h2_kappay2 = 0.05,
      h2_theta1  = 0.02, h2_theta2  = 0.02 ),
    beta1 = beta_true, beta2 = beta_true ) {
          ##### Initialize parameters #####
          ## We assume the sample size of tissue1 equals to that of tissue2
          ## GTEx(tissue1) = GTEx(tissue2); GWAS(1) = GWAS(2)
          nsample <- nGTEx + nGWAS;
          #---- Tissue 1 ----#
          delta1       <- rnorm( nsnp[1], 0, 1 );
          Sigma_kappa1 <- matrix( c( 1, r_kappa[ 1 ], r_kappa[ 1 ], 1 ), nrow = 2 );
          kappa1       <- mvrnorm( nsnp[ 1 ], mu = c( 0, 0 ), Sigma = Sigma_kappa1 );
          kappax1      <- kappa1[ , 1 ]; kappay1 <- kappa1[ , 2 ];
          theta1 <- rnorm( nsnp[ 1 ], 0, 1 )
          fai1 <- mvrnorm( nconf, mu = c( 0, 0 ), Sigma = matrix( c( 1, 0, 0, 1 ), nrow = 2 ) )
          faix1 <- fai1[ , 1 ]; faiy1 <- fai1[ , 2 ]
          G1 <- genotype_continuous( sample_size = nsample, nsnp = nsnp[1], r_LD = r_LD[1] ) %>% 
            genotype_to_factor( ., c( 0.05, 0.5 ) )
          U1 <- matrix( rnorm( nsample * nconf ), ncol = nconf )
          # dim(G1); dim(U1)
          #---- Tissue 2 -----#
          delta2 <- rnorm( nsnp[ 2 ], 0, 1 ); 
          Sigma_kappa2 <- matrix( c( 1, r_kappa[ 2 ], r_kappa[ 2 ], 1 ), nrow = 2 );
          kappa2 <- mvrnorm( nsnp[ 2 ], mu = c( 0, 0 ), Sigma = Sigma_kappa2 )
          kappax2 <- kappa2[ , 1 ]; kappay2 <- kappa2[ , 2 ];
          theta2 <- rnorm( nsnp[ 2 ], 0, 1 )
          fai2 <- mvrnorm( nconf, mu = c( 0, 0 ), Sigma = matrix( c( 1, 0, 0, 1 ), nrow = 2 ) )
          faix2 <- fai2[ , 1 ]; faiy2 <- fai2[ , 2 ]
          G2 <- genotype_continuous( sample_size = nsample, nsnp = nsnp[ 2 ], r_LD = r_LD[ 2 ] ) %>%
            genotype_to_factor( ., c( 0.05, 0.5 ) )
          U2 <- matrix( rnorm( nsample * nconf ), ncol = nconf )
          # dim(G2); dim(U2)
          
          ##### Normalize parameters #####
          h2_total <- sum( unlist( h2 ) )
          k_scale <- ( beta1^2 + beta2^2 + 1 )/( 1 - h2_total )
          #---- Tissue 1 -----
          G_delta1  <- G1 %*% delta1; G_kappax1 <- G1 %*% kappax1
          if( beta1 != 0 ){
            delta1   <- delta1 *sqrt( h2$h2_delta1 *k_scale/beta1^2/as.numeric( var( G_delta1  ) ) )
            kappax1  <- kappax1*sqrt( h2$h2_kappax1*k_scale/beta1^2/as.numeric( var( G_kappax1 ) ) )
            G_delta1 <- G1 %*% delta1; G_kappax1 <- G1 %*% kappax1
          }
          
          if( h2$h2_kappay1 == 0 ){
            G_kappay1 <- G1 %*% rep( 0, nsnp[1] )
          } else {
            G_kappay1 <- G1 %*% kappay1
            kappay1   <- kappay1*sqrt( h2$h2_kappay1*k_scale/as.numeric( var( G_kappay1 ) ) )
            G_kappay1 <- G1 %*% kappay1
          }
          
          if( h2$h2_theta1 == 0 ){
            G_theta1 <- G1 %*% rep( 0, nsnp[1] )
          } else {
            G_theta1 <- G1 %*% theta1
            theta1   <- theta1*sqrt( h2$h2_theta1*k_scale/as.numeric( var( G_theta1 ) ) )
            G_theta1 <- G1 %*% theta1
          }
          
          U_faix1   <- U1 %*% faix1; U_faix1 <- U_faix1 / as.numeric( sqrt( var( U_faix1 ) / 0.8 ) ) 
          U_faiy1   <- U1 %*% faiy1; U_faiy1 <- U_faiy1 / as.numeric( sqrt( var( U_faiy1 ) / 0.4 ) )
          X1 <- G_delta1 + G_kappax1 + U_faix1 + rnorm( nsample )*sqrt( 1 - as.numeric( var( U_faix1 ) ) )
          #---- Tissue 2 -----
          G_delta2  <- G2 %*% delta2; G_kappax2 <- G2 %*% kappax2
          if( beta2 != 0 ){
            delta2   <- delta2 *sqrt( h2$h2_delta2 *k_scale/beta2^2/as.numeric( var( G_delta2  ) ) )
            kappax2  <- kappax2*sqrt( h2$h2_kappax2*k_scale/beta2^2/as.numeric( var( G_kappax2 ) ) )
            G_delta2 <- G2 %*% delta2; G_kappax2 <- G2 %*% kappax2
          }
          
          if( h2$h2_kappay2 == 0 ){
            G_kappay2 <- G2 %*% rep( 0, nsnp[2] )
          } else {
            G_kappay2 <- G2 %*% kappay2
            kappay2   <- kappay2*sqrt( h2$h2_kappay2*k_scale/as.numeric( var( G_kappay2 ) ) )
            G_kappay2 <- G2 %*% kappay2
          }
          
          if( h2$h2_theta2 == 0 ){
            G_theta2 <- G2 %*% rep( 0, nsnp[2] )
          } else {
            G_theta2 <- G2 %*% theta2
            theta2   <- theta2*sqrt( h2$h2_theta2*k_scale/as.numeric( var( G_theta2 ) ) )
            G_theta2 <- G2 %*% theta2
          }
          
          U_faix2   <- U2 %*% faix2; U_faix2 <- U_faix2 / as.numeric( sqrt( var( U_faix2 ) / 0.8 ) ) 
          U_faiy2   <- U2 %*% faiy2; U_faiy2 <- U_faiy2 / as.numeric( sqrt( var( U_faiy2 ) / 0.4 ) )
          X2 <- G_delta2 + G_kappax2 + U_faix2 + rnorm( nsample )*sqrt( 1 - as.numeric( var( U_faix2 ) ) )
          
          Y <- beta1*X1 + beta2*X2 + G_kappay1 + G_kappay2 + G_theta1 + G_theta2 + U_faiy1 + U_faiy2 + rnorm( nsample )*as.numeric( sqrt( 1 - var( U_faiy1 ) - var( U_faiy2 ) ) )
          
          # var( beta1*G_delta1 ) / var( Y ); var( beta2*G_delta2 ) / var( Y )
          # var( beta1*G_kappax1 ) / var( Y ); var( beta2*G_kappax2 ) / var( Y )
          # var( G_kappay1 ) / var( Y ); var( G_kappay2 ) / var( Y )
          # var( G_theta1 ) / var( Y ); var( G_theta2 ) / var( Y )
          
          ##### Linear-regression #####
          model_gamma1 <- MUSE::lm_cpp( X1[ 1 : nGTEx ], G1[ 1 : nGTEx, ] )
          model_Gamma1 <- MUSE::lm_cpp( Y[ ( nGTEx + 1 ) : nsample ], G1[ ( nGTEx + 1 ) : nsample, ] )
          model_gamma2 <- MUSE::lm_cpp( X2[ 1 : nGTEx ], G2[ 1 : nGTEx, ] )
          model_Gamma2 <- MUSE::lm_cpp( Y[ ( nGTEx + 1 ) : nsample ], G2[ ( nGTEx + 1 ) : nsample, ] )
          
          mydata1 <- list( bx   = as.matrix( model_gamma1$coef ), 
                           bxse = as.matrix( model_gamma1$std ), 
                           by   = as.matrix( model_Gamma1$coef ), 
                           byse = as.matrix( model_Gamma1$std ),
                           GT = G1,
                           R_hat = cor( G1 ) )
          mydata2 <- list( bx   = as.matrix( model_gamma2$coef ), 
                           bxse = as.matrix( model_gamma2$std ),
                           by   = as.matrix( model_Gamma2$coef ), 
                           byse = as.matrix( model_Gamma2$std ),
                           GT = G2,
                           R_hat = cor( G2 ) )
          
          return( list( Tissue1 = mydata1, Tissue2 = mydata2 ) )
        }
        
        mydata <- data_generating(
          nGTEx = 10000, nGWAS = 10000,
          nsnp = c( 100, 100 ), nconf = 10,  ## nconf for number of confounders ##
          r_kappa = c( 0.2, 0.2 ),
          r_LD = c( 0.5, 0.5 ),
          h2 = list(
            h2_delta1  = h2_delta,  h2_delta2  = h2_delta,
            h2_kappax1 = h2_kappax, h2_kappax2 = h2_kappax,
            h2_kappay1 = h2_kappa, h2_kappay2 = h2_kappa,
            h2_theta1  = 0.02, h2_theta2  = 0.02 ),  ############ Check Point 4
          beta1 = beta_true, beta2 = beta_true )
        
        mrinput <- mr_input( bx   = as.numeric( mydata$Tissue1$bx ),
                             bxse = as.numeric( mydata$Tissue1$bxse ),
                             by   = as.numeric( mydata$Tissue1$by ),
                             byse = as.numeric( mydata$Tissue1$byse ) )
        
        res_median <- mr_median( mrinput )
        res_mode   <- mr_mbe( mrinput )
        res_ivw    <- mr_ivw( mrinput )
        res_pivw   <- mr_ivw( mrinput )
        res_egger  <- mr_egger( mrinput )
        res_conmix <- mr_conmix( mrinput )
        res_MULTI  <- MULTI( mydata )
        HD_matrix <- res_MULTI[[ 3 ]]
        
        tissue_similar_0.1 <- c()
        for( j in 1:length( mydata ) ){
          # j <- 1
          if( HD_matrix[ 1, j ] < 0.1 ) tissue_similar_0.1 <- c( tissue_similar_0.1, paste0( 'Tissue', j ) )
        }
        bx_new <- c()
        bxse_new <- c()
        by_new <- c()
        byse_new <- c()
        GT_new <- c()
        for( i in 1:length( tissue_similar_0.1 ) ){
          bx_new   <- rbind( bx_new,   mydata[[ tissue_similar_0.1[i] ]]$bx )
          bxse_new <- rbind( bxse_new, mydata[[ tissue_similar_0.1[i] ]]$bxse )  
          by_new   <- rbind( by_new,   mydata[[ tissue_similar_0.1[i] ]]$by )
          byse_new <- rbind( byse_new, mydata[[ tissue_similar_0.1[i] ]]$byse )
          GT_new   <- cbind( GT_new,   mydata[[ tissue_similar_0.1[i] ]]$GT )
        }
        mydata_new    <- list( bx = bx_new, bxse = bxse_new, by = by_new, byse = byse_new, R_hat = cor( GT_new ) )
        res_MULTI_0.1_final <- MULTI_single( mydata_new )
        
        tissue_similar_0.3 <- c()
        for( j in 1:length( mydata ) ){
          if( HD_matrix[ 1, j ] < 0.3 ) tissue_similar_0.3 <- c( tissue_similar_0.3, paste0( 'Tissue', j ) )
        }
        bx_new <- c()
        bxse_new <- c()
        by_new <- c()
        byse_new <- c()
        GT_new <- c()
        for( i in 1:length( tissue_similar_0.3 ) ){
          bx_new   <- rbind( bx_new,   mydata[[ tissue_similar_0.3[i] ]]$bx )
          bxse_new <- rbind( bxse_new, mydata[[ tissue_similar_0.3[i] ]]$bxse )  
          by_new   <- rbind( by_new,   mydata[[ tissue_similar_0.3[i] ]]$by )
          byse_new <- rbind( byse_new, mydata[[ tissue_similar_0.3[i] ]]$byse )
          GT_new   <- cbind( GT_new,   mydata[[ tissue_similar_0.3[i] ]]$GT )
        }
        mydata_new    <- list( bx = bx_new, bxse = bxse_new, by = by_new, byse = byse_new, R_hat = cor( GT_new ) )
        res_MULTI_0.3_final <- MULTI_single( mydata_new )
        
        tissue_similar_0.5 <- c()
        for( j in 1:length( mydata ) ){
          if( HD_matrix[ 1, j ] < 0.5 ) tissue_similar_0.5 <- c( tissue_similar_0.5, paste0( 'Tissue', j ) )
        }
        bx_new <- c()
        bxse_new <- c()
        by_new <- c()
        byse_new <- c()
        GT_new <- c()
        for( i in 1:length( tissue_similar_0.5 ) ){
          bx_new   <- rbind( bx_new,   mydata[[ tissue_similar_0.5[i] ]]$bx )
          bxse_new <- rbind( bxse_new, mydata[[ tissue_similar_0.5[i] ]]$bxse )  
          by_new   <- rbind( by_new,   mydata[[ tissue_similar_0.5[i] ]]$by )
          byse_new <- rbind( byse_new, mydata[[ tissue_similar_0.5[i] ]]$byse )
          GT_new   <- cbind( GT_new,   mydata[[ tissue_similar_0.5[i] ]]$GT )
        }
        mydata_new    <- list( bx = bx_new, bxse = bxse_new, by = by_new, byse = byse_new, R_hat = cor( GT_new ) )
        res_MULTI_0.5_final <- MULTI_single( mydata_new )
        
        e_median <- c( res_median@Estimate, res_median@CILower, res_median@CIUpper )
        e_mode   <- c( res_mode@Estimate,   res_mode@CILower,   res_mode@CIUpper )
        e_ivw    <- c( res_ivw@Estimate,  res_ivw@CILower,  res_ivw@CIUpper )
        e_pivw   <- c( res_pivw@Estimate, res_pivw@CILower, res_pivw@CIUpper )
        e_egger  <- c( res_egger@Estimate, res_egger@CILower.Est, res_egger@CIUpper.Est )
        e_conmix <- c( res_conmix@Estimate,  res_conmix@CILower, res_conmix@CIUpper )
        e_MULTI  <- c( res_MULTI[[1]]$mu_beta,
                       res_MULTI[[1]]$mu_beta - 1.96 * res_MULTI[[1]]$se_beta,
                       res_MULTI[[1]]$mu_beta + 1.96 * res_MULTI[[1]]$se_beta )
        e_MULTI_0.1 <- c( res_MULTI_0.1_final$mu_beta,
                          res_MULTI_0.1_final$mu_beta - 1.96 * res_MULTI_0.1_final$se_beta,
                          res_MULTI_0.1_final$mu_beta + 1.96 * res_MULTI_0.1_final$se_beta )
        e_MULTI_0.3 <- c( res_MULTI_0.3_final$mu_beta,
                          res_MULTI_0.3_final$mu_beta - 1.96 * res_MULTI_0.3_final$se_beta,
                          res_MULTI_0.3_final$mu_beta + 1.96 * res_MULTI_0.3_final$se_beta )
        e_MULTI_0.5 <- c( res_MULTI_0.5_final$mu_beta,
                          res_MULTI_0.5_final$mu_beta - 1.96 * res_MULTI_0.5_final$se_beta,
                          res_MULTI_0.5_final$mu_beta + 1.96 * res_MULTI_0.5_final$se_beta )
        ##### Type I error #####
        typeI_median    <- ifelse( e_median[2]    > 0 | e_median[3]    < 0, 'Reject', 'Accept' )
        typeI_mode      <- ifelse( e_mode[2]      > 0 | e_mode[3]      < 0, 'Reject', 'Accept' )
        typeI_ivw       <- ifelse( e_ivw[2]       > 0 | e_ivw[3]       < 0, 'Reject', 'Accept' )
        typeI_pivw      <- ifelse( e_pivw[2]      > 0 | e_pivw[3]      < 0, 'Reject', 'Accept' )
        typeI_egger     <- ifelse( e_egger[2]     > 0 | e_egger[3]     < 0, 'Reject', 'Accept' )
        typeI_conmix    <- ifelse( e_conmix[2]    > 0 | e_conmix[3]    < 0, 'Reject', 'Accept' )
        typeI_MULTI     <- ifelse( e_MULTI[2]     > 0 | e_MULTI[3]     < 0, 'Reject', 'Accept' )
        typeI_MULTI_0.1 <- ifelse( e_MULTI_0.1[2] > 0 | e_MULTI_0.1[3] < 0, 'Reject', 'Accept' )     
        typeI_MULTI_0.3 <- ifelse( e_MULTI_0.3[2] > 0 | e_MULTI_0.3[3] < 0, 'Reject', 'Accept' ) 
        typeI_MULTI_0.5 <- ifelse( e_MULTI_0.5[2] > 0 | e_MULTI_0.5[3] < 0, 'Reject', 'Accept' ) 
        typeIerror <- c( typeI_median, typeI_mode, typeI_ivw, typeI_pivw, typeI_egger, typeI_conmix,
                         typeI_MULTI, typeI_MULTI_0.1, typeI_MULTI_0.3, typeI_MULTI_0.5 )
        names( typeIerror ) <- c( 'Median', 'Mode', 'IVW', 'pIVW', 'Egger', 'Conmix',
                                  'MULTI', 'MULTI_0.1', 'MULTI_0.3', 'MULTI_0.5' )
        
        ##### Bias #####
        bias_median    <- e_median[1]    - beta_true
        bias_mode      <- e_mode[1]      - beta_true
        bias_ivw       <- e_ivw[1]       - beta_true
        bias_pivw      <- e_pivw[1]      - beta_true
        bias_egger     <- e_egger[1]     - beta_true
        bias_conmix    <- e_conmix[1]    - beta_true
        bias_MULTI     <- e_MULTI[1]     - beta_true
        bias_MULTI_0.1 <- e_MULTI_0.1[1] - beta_true
        bias_MULTI_0.3 <- e_MULTI_0.3[1] - beta_true
        bias_MULTI_0.5 <- e_MULTI_0.5[1] - beta_true
        bias        <- c( bias_median, bias_mode, bias_ivw, bias_pivw, bias_egger, bias_conmix,
                          bias_MULTI, bias_MULTI_0.1, bias_MULTI_0.3, bias_MULTI_0.5 )
        names( bias ) <- c( 'Median', 'Mode', 'IVW', 'pIVW', 'Egger', 'Conmix',
                            'MULTI', 'MULTI_0.1', 'MULTI_0.3', 'MULTI_0.5' )
        
        ##### Merge Rate #####
        merge_rate <- paste0( c( tissue_similar_0.1, tissue_similar_0.3, tissue_similar_0.5 ), collapse = ' ' )
        names( merge_rate ) <- 'Merge_Rate'
        result <- list( typeIerror = typeIerror, bias = bias, merge_rate = merge_rate )
        
        return( result )
        
      }   ### Check 5
      
      a <- mclapply( ( 100*( k - 1 ) + 1 ):( 100*k ), function( m ) simulate_once( m, h2_kappa, h2_gamma ), mc.cores = detectCores( ) - 1  )   ### Check 6
      return( a )
      
    },
    args = list( k, h2_kappa, h2_gamma )   ### Check 7
  )
  
  start_time <- Sys.time( )
  while( rp$is_alive( ) ){
    
    elapsed <- difftime( Sys.time( ), start_time, units = 'secs' )
    
    if( elapsed > 300 ){
      
      rp$kill( )
      return( NULL )
      
    }
    
    Sys.sleep( 10 )
    
  }
  
  return( rp$get_result( ) )
  
}


h2_kappa_TBD <- c( 0, 0.01, 0.02, 0.03, 0.04, 0.05 )  ### Check 7
h2_gamma_TBD <- c( 0, 10^-2, 10^-1 )

for( i in 1:length( h2_kappa_TBD ) ){   ### Check 8
  
  for( j in 1:length( h2_gamma_TBD ) ){
    
    h2_kappa <- h2_kappa_TBD[ i ]   ### Check 9
    h2_gamma <- h2_gamma_TBD[ j ]
    
    sim_res <- list( )
    success_count <- 0
    k <- 1
    
    while( success_count < 5 ){
      print( k )
      
      sim_res <- run_task( k, h2_kappa, h2_gamma )   ### Check 10
      
      if( !is.null( sim_res ) ) {
        save( sim_res, file = paste0( 'result_', i, '_', j, '_', success_count + 1, '_h2theta.RData' ) )  # Check 11
        success_count <- success_count + 1
      }
      
      k <- k + 1
      gc()
      
    }
    
    
  }
  
}

