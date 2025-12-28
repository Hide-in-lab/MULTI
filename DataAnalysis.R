#################### < Step 1 > Extract all available SNPs for each chromosome ####################
rs_all <- list( )
for( i in 1:22 ){
  gds_path <-  paste0( '/data1/Chengyu/ReferencePanel/ALL.chr', i,
                       '.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.gds')
  genofile <- snpgdsOpen( gds_path, readonly = TRUE )
  # rs_all   <- read.gdsn( index.gdsn( genofile, 'snp.id' ) )
  rs_all[[ i ]] <- read.gdsn( index.gdsn( genofile, 'snp.id' ) )
  # save( rs_all, file = paste0( '/data1/Chengyu/ReferencePanel/rs_chr', i, '.RData' ) )
}
names( rs_all ) <- paste0( 'rs_chr', 1:22 )
# # save( rs_all, file = paste0( '/data1/Chengyu/ReferencePanel/rs_all.RData' ) )

#################### < Step 2 > Load Data & Data Preprocess ####################
setwd( '~/CY/Data' )
datapath_gtex <- 'GTEx/eQTL/'
datapath_gwas <- 'GWAS/Depression/'
library( data.table )
library( dplyr )
library( SNPRelate )
library( Matrix )
library( parallel )
########## Load data ##########
##### Colon #####
myfiles <- list.files( datapath_gtex )[ c( 46, 48 ) ]
colon_sigmoid <- fread( paste0( datapath_gtex, myfiles[ 1 ] ) )
colon_trans   <- fread( paste0( datapath_gtex, myfiles[ 2 ] ) )
depression    <- fread( file = paste0( datapath_gwas, 'GCST90477004.tsv.gz') )
head( colon_sigmoid ); head( colon_trans ); head( depression )
# crc <- fread( file = paste0( datapath_gwas, '20001_1020.v1.0.fastGWA.gz') )
depression$beta <- sapply( depression$odds_ratio, log )
dt <- as.data.table( depression )
dt[ , se := abs( beta / qnorm( 1 - p_value / 2 ) ) ]
dt <- dt[ !is.na( dt$se ) & !is.infinite( dt$se ), ]
########## Subset & Preprocess data ##########
colon_trans_small   <- subset( colon_trans,   pval_nominal < 5e-8 ) %>% select( 'CHR', 'ID', 'rs_ID', 'slope', 'slope_se' )
colon_sigmoid_small <- subset( colon_sigmoid, pval_nominal < 5e-8 ) %>% select( 'CHR', 'ID', 'rs_ID', 'slope', 'slope_se' )
depression_small    <- dt %>% select( 'rsid', 'beta', 'se' )
colnames( colon_trans_small )   <- c( 'chr', 'gene', 'rs', 'bx', 'bxse' )
colnames( colon_sigmoid_small ) <- c( 'chr', 'gene', 'rs', 'bx', 'bxse' )
colnames( depression_small )    <- c( 'rs',   'by', 'byse' )
colon_trans_small$chr   <- sub("^.*chr", "", colon_trans_small$chr )
colon_sigmoid_small$chr <- sub("^.*chr", "", colon_sigmoid_small$chr )
colon_trans_small   <- subset( colon_trans_small,   chr %in% 1:22 )
colon_sigmoid_small <- subset( colon_sigmoid_small, chr %in% 1:22 )
# head( colon_sigmoid_small ); head( colon_trans_small ); head( depression_small )
genes_common <- intersect( unique( colon_sigmoid_small$gene ), unique( colon_trans_small$gene ) )
gene_info <- unique( select( colon_trans_small, 'chr', 'gene' ) ) %>%
  as.data.frame( ) %>%
  subset(., gene %in% genes_common )
gene_info$chr <- as.numeric( gene_info$chr )
rm( colon_trans ); rm( colon_sigmoid ); rm( depression ); rm( dt ); gc()

#################### < Step 3 > Write Gene Information ####################
############### < Step 3.1 > Set colon_trans as primary tissue ###############
gene_info_trans <- list( )
gene_info_trans <- mclapply( 1:nrow( gene_info ), function( i ){
  
  target_chr  <- gene_info[ i, 1 ]
  target_gene <- gene_info[ i, 2 ]
  
  trans_int   <- colon_trans_small   %>% filter( gene == target_gene )
  sigmoid_int <- colon_sigmoid_small %>% filter( gene == target_gene )
  ### Modify here ###
  rs_trans   <- unique( trans_int$rs ) %>% 
    .[ . %in%  depression_small$rs ] %>%
    .[ . %in% rs_all[[ target_chr ]] ]
  rs_sigmoid <- unique( sigmoid_int$rs ) %>% 
    .[ . %in%  depression_small$rs ] %>%
    setdiff( ., rs_trans ) %>%
    .[ . %in% rs_all[[ target_chr ]] ]
  ### Tissue 1, Set colon_trans as primary tissue ###
  Tissue1 <- depression_small %>% filter( rs %in% rs_trans )
  Tissue1 <- merge(Tissue1, trans_int %>% filter(rs %in% rs_trans), by = 'rs')
  ### Tissue 2, Set colon_sigmoid as secondary tissue ###
  Tissue2 <- depression_small %>% filter(rs %in% rs_sigmoid)
  Tissue2 <- merge(Tissue2, sigmoid_int %>% filter(rs %in% rs_sigmoid), by = 'rs')
  
  result <- list( gene = target_gene, chr = target_chr, nSNP1 = nrow( Tissue1 ), nSNP2 = nrow( Tissue2 ) )
  
  return( result )
  
}, mc.cores = 10 )
# unlist( gene_info_trans ) %>% matrix( byrow = T, ncol = 4)
# save( gene_info_trans, file = '/data1/Chengyu/MULTI/DataAnalysis/gene_info_trans_sigmoid_depression.RData' )


############### < Step 3.2 > Set colon_sigmoid as primary tissue ###############
gene_info_sigmoid <- list( )
gene_info_sigmoid <- mclapply( 1:nrow( gene_info ), function( i ){
  
  target_chr  <- gene_info[ i, 1 ]
  target_gene <- gene_info[ i, 2 ]
  
  trans_int   <- colon_trans_small   %>% filter( gene == target_gene )
  sigmoid_int <- colon_sigmoid_small %>% filter( gene == target_gene )
  
  ### Modify here ###
  rs_sigmoid <- unique( sigmoid_int$rs ) %>% 
    .[ . %in%  depression_small$rs ] %>%
    .[ . %in% rs_all[[ target_chr ]] ]
  rs_trans   <- unique( trans_int$rs ) %>% 
    .[ . %in%  depression_small$rs ] %>%
    setdiff( ., rs_sigmoid ) %>%
    .[ . %in% rs_all[[ target_chr ]] ]
  
  ### Tissue 1, Set colon_sigmoid as primary tissue ###
  Tissue1 <- depression_small %>% filter( rs %in% rs_sigmoid )
  Tissue1 <- merge( Tissue1, sigmoid_int %>% filter( rs %in% rs_sigmoid ), by = 'rs' )
  ### Tissue 2, Set colon_trans as secondary tissue ###
  Tissue2 <- depression_small %>% filter( rs %in% rs_trans )
  Tissue2 <- merge( Tissue2, trans_int %>% filter( rs %in% rs_trans ), by = 'rs' )
  
  result <- list( gene = target_gene, chr = target_chr, nSNP1 = nrow( Tissue1 ), nSNP2 = nrow( Tissue2 ) )
  
  return( result )
  
}, mc.cores = 10 )
# unlist( gene_info_sigmoid ) %>% matrix( byrow = T, ncol = 4)
# save( gene_info_sigmoid, file = '/data1/Chengyu/MULTI/DataAnalysis/gene_info_sigmoid_trans_depression.RData' )
rm( gene_info ); gc()

#################### < Step 4 > Causal Inference for colon transverse/sigmoid with Depression ####################
pattern_drop <- '^(RP\\d+|AC\\d+|AL\\d+|AP\\d+|CTD-|CTA-|FAM\\d+|LOC\\d+|LINC\\d+|XXbac-|MIR|SNORA|SNORD|SCARNA|RNU|RNA|RN7SL)'
load( 'ReferencePanel/rs_all.RData' )
##### Customized function #####
force2full <- function( x ){

  library( Matrix )
  is_full <- rankMatrix( x )[ 1 ] == nrow( x )
  
  if( is_full ){
    
    return( x )

  } else{

    x <- as.matrix( nearPD( x, corr = TRUE)$mat )
    return( x )
    
  }

}

sample_Eur <- fread( 'ReferencePanel/integrated_call_samples_v3.20130502.ALL.panel',
                     header = T, fill = T ) %>% 
  as.data.frame( . ) %>% subset( ., super_pop == 'EUR' ) %>% .$sample

Rcpp::sourceCpp( '~/CY/MULTI/MULTI_Final.cpp' )
############### < Step 4.1 > Causal inference for colon trans ###############
load( '~/CY/MULTI/DataAnalysis/gene_info_trans_sigmoid_depression.RData' )
gene_info_trans <- unlist( gene_info_trans ) %>%
  matrix( byrow = T, ncol = 4 ) %>% 
  as.data.frame( )
colnames( gene_info_trans ) <- c( 'gene', 'chr', 'nSNP1', 'nSNP2' )
gene_info_trans$nSNP1 <- as.numeric( gene_info_trans$nSNP1 )
gene_info_trans$nSNP2 <- as.numeric( gene_info_trans$nSNP2 )
genes_TBD <- subset( gene_info_trans, gene_info_trans$nSNP1 >= 20 & gene_info_trans$nSNP2 >= 20 )$gene
genes_TBD <- genes_TBD[ !grepl(pattern_drop, genes_TBD, ignore.case = FALSE) ]
gene_info_trans <- subset( gene_info_trans, gene %in% genes_TBD ) %>%
  .[ !duplicated( .$gene ), ]

start_time <- Sys.time( )
result_trans <- list( )
for( i in 1:nrow( gene_info_trans ) ){
# for( i in 1:5 ){
    # i <- 1
  tryCatch( 
    {

      target_gene <- gene_info_trans[ i, 1 ]
      target_chr  <- as.numeric( gene_info_trans[ i, 2 ] )

      trans_int   <- colon_trans_small   %>% filter( gene == target_gene )
      sigmoid_int <- colon_sigmoid_small %>% filter( gene == target_gene )
      ### Modify here ###
      rs_trans   <- unique( trans_int$rs ) %>% 
        .[ . %in%  depression_small$rs ] %>%
        .[ . %in% rs_all[[ target_chr ]] ]
      rs_sigmoid <- unique( sigmoid_int$rs ) %>% 
        .[ . %in%  depression_small$rs ] %>%
        setdiff( ., rs_trans ) %>%
        .[ . %in% rs_all[[ target_chr ]] ]
      ### Tissue 1, Set colon_trans as primary tissue ###
      Tissue1 <- depression_small %>% filter( rs %in% rs_trans )
      Tissue1 <- merge(Tissue1, trans_int %>% filter(rs %in% rs_trans), by = 'rs')
      ### Tissue 2, Set colon_sigmoid as secondary tissue ###
      Tissue2 <- depression_small %>% filter(rs %in% rs_sigmoid)
      Tissue2 <- merge(Tissue2, sigmoid_int %>% filter(rs %in% rs_sigmoid), by = 'rs')
      
      gds_path <-  paste0( 'ReferencePanel/ALL.chr',
                           target_chr,
                           '.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.gds')
      
      genofile <- snpgdsOpen( gds_path, readonly = TRUE )
      ##### Calculate R_gat #####
      R_hat_trans   <- snpgdsGetGeno( genofile, sample.id = sample_Eur, snp.id = rs_trans   ) %>%
        cor() %>% force2full( )
      R_hat_sigmoid <- snpgdsGetGeno( genofile, sample.id = sample_Eur, snp.id = rs_sigmoid ) %>%
        cor() %>% force2full( )
      R_hat <- snpgdsGetGeno( genofile, sample.id = sample_Eur, snp.id = c( rs_trans, rs_sigmoid ) ) %>%
        cor() %>% force2full( )
      closefn.gds( genofile )
      
      ##### Prepare Input #####
      mydata <- list(  )
      mydata[[ 1 ]] <- list( bx   = as.matrix( Tissue1$bx ) + rnorm( length( rs_trans ), 0, 0.00001 ),
                             bxse = as.matrix( Tissue1$bxse ) + 1/rgamma( length( rs_trans ), 10, rate = 0.0001 ),
                             by   = as.matrix( Tissue1$by ) + rnorm( length( rs_trans ), 0, 0.00001 ),
                             byse = as.matrix( Tissue1$byse ) + 1/rgamma( length( rs_trans ), 10, rate = 0.0001 ),
                             R_hat = R_hat_trans )
      mydata[[ 2 ]] <- list( bx   = as.matrix( Tissue2$bx ) + rnorm( length( rs_sigmoid ), 0, 0.000001 ),
                             bxse = as.matrix( Tissue2$bxse ) + 1/rgamma( length( rs_sigmoid ), 10, rate = 0.0001 ),
                             by   = as.matrix( Tissue2$by ) + rnorm( length( rs_sigmoid ), 0, 0.000001 ),
                             byse = as.matrix( Tissue2$byse ) + 1/rgamma( length( rs_sigmoid ), 10, rate = 0.0001 ),
                             R_hat = R_hat_sigmoid )
      names( mydata ) <- c( 'Tissue1', 'Tissue2' )
      mydata_new  <- list( bx   = as.matrix( c( mydata[[ 1 ]]$bx, mydata[[ 2 ]]$bx ) ),
                           bxse = as.matrix( c( mydata[[ 1 ]]$bxse, mydata[[ 2 ]]$bxse ) ),
                           by   = as.matrix( c( mydata[[ 1 ]]$by, mydata[[ 2 ]]$by ) ),
                           byse = as.matrix( c( mydata[[ 1 ]]$byse, mydata[[ 2 ]]$byse ) ),
                           R_hat = R_hat )

      ##### Calculate Results #####
      # res <- MULTI::MULTI( mydata, r2 = 0 )
      res <- MULTI( mydata, r2 = 0 )
      ##### No integration #####
      res_MULTI <- res$`Estimands for Tissue 1`
      HD_matrix <- res$`Hellinger Distance Matrix`
      ##### Cutoff = 0.1 #####
      res_int <- MULTI_single( mydata_new, r2 = 0 )

      result_trans[[ i ]] <- list( gene = target_gene, res_MULTI = res_MULTI, HD = HD_matrix[ 1, 2 ], res_int = res_int )

    }, error = function(e) {
      
      result_trans[[ i ]] <- list( gene = NA, res_MULTI = NA, HD = NA, res_int = NA )

    }
 )

  print( i )
  
}
end_time <- Sys.time( )
elapsed <- end_time - start_time
# # save( result_trans, file = '~/CY/MULTI/DataAnalysis/result_trans.RData' )
load( '~/CY/MULTI/DataAnalysis/result_trans.RData' )
a <- unlist( result_trans ) %>% matrix( byrow = T, ncol = 6 ) %>% as.data.frame( )
table( a$V2 - 1.96*a$V3 > 0 |  a$V2 + 1.96*a$V3 < 0 )
a[ , 2:6 ] <- apply( a[ , 2:6 ], 2, as.numeric )
table( a$V4 < 0.5 )
summary( a$V4 )
plot( density( a$V4 ) )
############### < Step 4.2 > Causal inference for colon trans ###############
load( '~/CY/MULTI/DataAnalysis/gene_info_sigmoid_trans_depression.RData' )
gene_info_sigmoid <- unlist( gene_info_sigmoid ) %>%
  matrix( byrow = T, ncol = 4) %>% 
  as.data.frame( )
colnames( gene_info_sigmoid ) <- c( 'gene', 'chr', 'nSNP1', 'nSNP2' )
gene_info_sigmoid$nSNP1 <- as.numeric( gene_info_sigmoid$nSNP1 )
gene_info_sigmoid$nSNP2 <- as.numeric( gene_info_sigmoid$nSNP2 )
genes_TBD <- subset( gene_info_sigmoid, gene_info_sigmoid$nSNP1 >= 20 & gene_info_sigmoid$nSNP2 >= 20 )$gene
genes_TBD <- genes_TBD[ !grepl(pattern_drop, genes_TBD, ignore.case = FALSE) ]
gene_info_sigmoid <- subset( gene_info_sigmoid, gene %in% genes_TBD ) %>%
  .[ !duplicated( .$gene ), ]

start_time <- Sys.time( )
result_sigmoid <- list( )
for( i in 1:nrow( gene_info_sigmoid ) ){

  tryCatch( 
    {

      target_gene <- gene_info_sigmoid[ i, 1 ]
      target_chr  <- as.numeric( gene_info_sigmoid[ i, 2 ] )
      
      sigmoid_int <- colon_sigmoid_small %>% filter( gene == target_gene )
      trans_int   <- colon_trans_small   %>% filter( gene == target_gene )
      
      ### Modify here ###
      rs_sigmoid <- unique( sigmoid_int$rs ) %>% 
        .[ . %in%  depression_small$rs ] %>%
        .[ . %in% rs_all[[ target_chr ]] ]
      rs_trans   <- unique( trans_int$rs ) %>% 
        .[ . %in%  depression_small$rs ] %>%
        setdiff( ., rs_sigmoid ) %>%
        .[ . %in% rs_all[[ target_chr ]] ]
      
      ### Tissue 1, Set colon_sigmoid as primary tissue ###
      Tissue1 <- depression_small %>% filter(rs %in% rs_sigmoid)
      Tissue1 <- merge( Tissue1, sigmoid_int %>% filter(rs %in% rs_sigmoid), by = 'rs' )
      ### Tissue 2, Set colon_trans as secondary tissue ###
      Tissue2 <- depression_small %>% filter( rs %in% rs_trans )
      Tissue2 <- merge( Tissue2, trans_int %>% filter(rs %in% rs_trans), by = 'rs' )
      
      gds_path <-  paste0( 'ReferencePanel/ALL.chr',
                           target_chr,
                           '.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.gds')

      genofile <- snpgdsOpen( gds_path, readonly = TRUE )
      ##### Calculate R_gat #####
      R_hat_sigmoid <- snpgdsGetGeno( genofile, sample.id = sample_Eur, snp.id = rs_sigmoid ) %>%
        cor() %>% force2full( )
      R_hat_trans   <- snpgdsGetGeno( genofile, sample.id = sample_Eur, snp.id = rs_trans   ) %>%
        cor() %>% force2full( )
      R_hat <- snpgdsGetGeno( genofile, sample.id = sample_Eur, snp.id = c( rs_sigmoid, rs_trans ) ) %>%
        cor() %>% force2full( )
      closefn.gds( genofile )
      
      ##### Prepare Input #####
      mydata <- list(  ) 
      mydata[[ 1 ]] <- list( bx   = as.matrix( Tissue1$bx ) + rnorm( length( rs_sigmoid ), 0, 0.000001 ),
                             bxse = as.matrix( Tissue1$bxse ) + 1/rgamma( length( rs_sigmoid ), 10, rate = 0.0001 ),
                             by   = as.matrix( Tissue1$by ) + rnorm( length( rs_sigmoid ), 0, 0.000001 ),
                             byse = as.matrix( Tissue1$byse ) + 1/rgamma( length( rs_sigmoid ), 10, rate = 0.0001 ),
                             R_hat = R_hat_sigmoid )
      mydata[[ 2 ]] <- list( bx   = as.matrix( Tissue2$bx ) + rnorm( length( rs_trans ), 0, 0.00001 ),
                             bxse = as.matrix( Tissue2$bxse ) + 1/rgamma( length( rs_trans ), 10, rate = 0.0001 ),
                             by   = as.matrix( Tissue2$by ) + rnorm( length( rs_trans ), 0, 0.00001 ),
                             byse = as.matrix( Tissue2$byse ) + 1/rgamma( length( rs_trans ), 10, rate = 0.0001 ),
                             R_hat = R_hat_trans )
      
      names( mydata ) <- c( 'Tissue1', 'Tissue2' )
      mydata_new  <- list( bx   = as.matrix( c( mydata[[ 1 ]]$bx, mydata[[ 2 ]]$bx ) ),
                           bxse = as.matrix( c( mydata[[ 1 ]]$bxse, mydata[[ 2 ]]$bxse ) ),
                           by   = as.matrix( c( mydata[[ 1 ]]$by, mydata[[ 2 ]]$by ) ),
                           byse = as.matrix( c( mydata[[ 1 ]]$byse, mydata[[ 2 ]]$byse ) ),
                           R_hat = R_hat )

      ##### Calculate Results #####
      # res <- MULTI::MULTI( mydata, r2 = 0 )
      res <- MULTI( mydata, r2 = 0 )
      ##### No integration #####
      res_MULTI <- res$`Estimands for Tissue 1`
      HD_matrix <- res$`Hellinger Distance Matrix`
      ##### Cutoff = 0.1 #####
      res_int <- MULTI_single( mydata_new, r2 = 0 )

      result_sigmoid[[ i ]] <- list( gene = target_gene, res_MULTI = res_MULTI, HD = HD_matrix[ 1, 2 ], res_int = res_int )

    }, error = function(e) {

      result_sigmoid[[ i ]] <- list( gene = NA, res_MULTI = NA, HD = NA, res_int = NA )

    }
  )

  print( i )
  
}
end_time <- Sys.time( )
elapsed <- end_time - start_time
# # save( result_sigmoid, file = '~/CY/MULTI/DataAnalysis/result_sigmoid.RData' )

#################### < Step 5 > Visualize the results ####################
############### Prepare Data ###############
library( rtracklayer )
library( ggplot2 )
load( '~/CY/MULTI/DataAnalysis/result_trans.RData' )
load( '~/CY/MULTI/DataAnalysis/result_sigmoid.RData' )
# gtf <- rtracklayer::import( 'gencode.v19.annotation.gtf.gz' )
chromosome_lengths <- c( 248956422, 242193529, 198295559, 190214555, 181538259,
                         170805979, 159345973, 145138636, 138394717, 133797422,
                         135086622, 133275309, 114364328, 107043718, 101991189,
                         90338345,  83257441,  80373285,  58617616,  64444167,
                         46709983,  50818468 )
cumulative_positions <- cumsum( chromosome_lengths )
chr_axis <- data.frame(
  chr   = paste0( 'chr', 1:22 ),
  start = c( 1, head( cumulative_positions, -1 ) + 1 ),
  end   = cumulative_positions )
############### Process Data (For colon trans) ###############
result_trans <- unlist( result_trans ) %>%
  matrix( byrow = T, ncol = 6 ) %>% 
  as.data.frame( )
result_trans[ , 2:6 ] <- apply( result_trans[ , 2:6 ], 2, as.numeric )
colnames( result_trans ) <- c( 'gene', 'beta', 'se', 'HD', 'beta_MULTI', 'se_MULTI' )
p <- c()
for( i in 1:nrow( result_trans ) ){
  p[ i ] <- ifelse( result_trans$beta[ i ] > 0, 
                    pnorm( 0, result_trans$beta[ i ], result_trans$se[ i ], lower.tail = T ),
                    pnorm( 0, result_trans$beta[ i ], result_trans$se[ i ], lower.tail = F ) )
}
result_trans$p <- p
p_MULTI <- c()
for( i in 1:nrow( result_trans ) ){
  p_MULTI[ i ] <- ifelse( result_trans$beta_MULTI[ i ] > 0, 
                          pnorm( 0, result_trans$beta_MULTI[ i ], result_trans$se_MULTI[ i ], lower.tail = T ),
                          pnorm( 0, result_trans$beta_MULTI[ i ], result_trans$se_MULTI[ i ], lower.tail = F ) )
}
result_trans$p_MULTI <- p_MULTI
gtf_small <- gtf[ gtf$type == 'gene' ]
genes_valid <- mcols( gtf_small )[[ 'gene_name' ]] %>%
  .[ mcols( gtf_small )[[ 'gene_name' ]]  %in% result_trans$gene ] %>%
  unique( )
result_trans <- subset( result_trans, gene %in% genes_valid )
gtf_small <- gtf_small[ gtf_small$gene_name %in% genes_valid, ] %>%
  .[ !duplicated( .$gene_name ), ]
mydata_trans <- data_frame(
  gene = genes_valid,
  chr  = as.character( seqnames( gtf_small ) ),
  strat = start( gtf_small ),
  HD = result_trans$HD,
  p = -log( result_trans$p, 10 ),
  p_sig = ifelse( result_trans$p < 0.05, '+', '-' ),
  p_MULTI = -log( result_trans$p_MULTI, 10 ),
  p_MULTI_sig = ifelse( result_trans$p_MULTI < 0.05, '+', '-' ),
  tissue = 'colon_trans'
)
mydata_trans$chr <- factor( mydata_trans$chr, levels = paste0( 'chr', 1:22 ) )
mydata_trans$pos <- NA
for( i in 1:nrow( mydata_trans ) ){
  mydata_trans$pos[i] <- ifelse( mydata_trans$chr[i] == 'chr1', mydata_trans$strat[i],
                           mydata_trans$strat[i] + chr_axis[ which( chr_axis$chr == mydata_trans$chr[i] ) - 1, ]$end )}
############### Process Data (For colon sigmoid) ###############
result_sigmoid <- unlist( result_sigmoid ) %>% 
                   matrix( byrow = T, ncol = 6 ) %>% 
                    as.data.frame( )
result_sigmoid[ , 2:6 ] <- apply( result_sigmoid[ , 2:6 ], 2, as.numeric )
colnames( result_sigmoid ) <- c( 'gene', 'beta', 'se', 'HD', 'beta_MULTI', 'se_MULTI' )
p <- c()
for( i in 1:nrow( result_sigmoid ) ){
  p[ i ] <- ifelse( result_sigmoid$beta[ i ] > 0, 
             pnorm( 0, result_sigmoid$beta[ i ], result_sigmoid$se[ i ], lower.tail = T ),
              pnorm( 0, result_sigmoid$beta[ i ], result_sigmoid$se[ i ], lower.tail = F ) )
}
result_sigmoid$p <- p
p_MULTI <- c()
for( i in 1:nrow( result_sigmoid ) ){
  p_MULTI[ i ] <- ifelse( result_sigmoid$beta_MULTI[ i ] > 0, 
                    pnorm( 0, result_sigmoid$beta_MULTI[ i ], result_sigmoid$se_MULTI[ i ], lower.tail = T ),
                    pnorm( 0, result_sigmoid$beta_MULTI[ i ], result_sigmoid$se_MULTI[ i ], lower.tail = F ) )
}
result_sigmoid$p_MULTI <- p_MULTI
# table( result_sigmoid$p < 0.05 ); table( result_sigmoid$p_MULTI < 0.05 )
# table( result_sigmoid$p < 0.05, result_sigmoid$p_MULTI < 0.05 )
gtf_small <- gtf[ gtf$type == 'gene' ]
genes_valid <- mcols( gtf_small )[[ 'gene_name' ]] %>%
                .[ mcols( gtf_small )[[ 'gene_name' ]]  %in% result_sigmoid$gene ] %>%
                 unique( )
result_sigmoid <- subset( result_sigmoid, gene %in% genes_valid )
gtf_small <- gtf_small[ gtf_small$gene_name %in% genes_valid, ] %>%
              .[ !duplicated( .$gene_name ), ]
mydata_sigmoid <- data_frame(
  gene = genes_valid,
  chr  = as.character( seqnames( gtf_small ) ),
  strat = start( gtf_small ),
  HD = result_sigmoid$HD,
  p = -log( result_sigmoid$p, 10 ),
  p_sig = ifelse( result_sigmoid$p < 0.05, '+', '-' ),
  p_MULTI = -log( result_sigmoid$p_MULTI, 10 ),
  p_MULTI_sig = ifelse( result_sigmoid$p_MULTI < 0.05, '+', '-' ),
  tissue = 'colon_sigmoid'
)
mydata_sigmoid$chr <- factor( mydata_sigmoid$chr, levels = paste0( 'chr', 1:22 ) )
mydata_sigmoid$pos <- NA
for( i in 1:nrow( mydata_sigmoid ) ){
  mydata_sigmoid$pos[i] <- ifelse( mydata_sigmoid$chr[i] == 'chr1', mydata_sigmoid$strat[i],
                           mydata_sigmoid$strat[i] + chr_axis[ which( chr_axis$chr == mydata_sigmoid$chr[i] ) - 1, ]$end )}
mydata_sigmoid$pos <- -mydata_sigmoid$pos
mydata <- rbind( mydata_trans, mydata_sigmoid )
mydata <- mydata[ !is.infinite( mydata$p_MULTI ) & !is.infinite( mydata$p ), ]
mydata$p_final <- ifelse( mydata$p_MULTI_sig == '+', mydata$p_MULTI,
                   ifelse( mydata$p_MULTI_sig == '-' & mydata$p_sig == '-' & mydata$HD <= 0.5, mydata$p,
                    ifelse( mydata$p_MULTI_sig == '-' & mydata$p_sig == '-' & mydata$HD > 0.5, mydata$p_MULTI,
                     ifelse( mydata$p_MULTI_sig == '-' & mydata$p_sig == '+' & mydata$HD <= 0.5, mydata$p, mydata$p_MULTI ) ) ) )
mydata$p_sig_final <- ifelse( mydata$p_MULTI_sig == '+', '+',
                       ifelse( mydata$p_MULTI_sig == '-' & mydata$p_sig == '-', '-', '+' ) )
mydata$int <- ifelse( mydata$p_MULTI_sig == '+', 'Single',
               ifelse( mydata$p_MULTI_sig == '-' & mydata$p_sig == '-' & mydata$HD <= 0.5, 'Multi',
                ifelse( mydata$p_MULTI_sig == '-' & mydata$p_sig == '-' & mydata$HD > 0.5, 'Single',
                 ifelse( mydata$p_MULTI_sig == '-' & mydata$p_sig == '+' & mydata$HD <= 0.5, 'Multi', 'Single' ) ) ) )
mydata_col <- data.frame(
  start = c( chr_axis$start, -chr_axis$start),
  end   = c( chr_axis$end, -chr_axis$end ),
  Freq  = as.numeric( c( table( subset( mydata, tissue == 'colon_trans' & p_sig_final == '+' )$chr ),
                         table( subset( mydata, tissue == 'colon_sigmoid' & p_sig_final == '+' )$chr ) ) ) )
# # write.csv( mydata, file = '~/CY/MULTI/DataAnalysis/mydata.csv' )
# # write.csv( mydata_col, file = '~/CY/MULTI/DataAnalysis/mydata_col.csv' )

p1 <- ggplot( data = mydata, aes( x = p_final, y = pos, colour = p_sig_final, shape = int ) ) +
  geom_point( size = 1.5 ) +
  labs( x = '-log(P value)', y = 'Chromosome', title = 'Summary results' ) +
  scale_y_continuous( expand = c( 0.01, 0 ), 
                      breaks = c( chr_axis$end, -chr_axis$end ), labels = rep( 1:22, 2) ) +
  scale_color_manual( values = c( '+' = '#D55E00', '-' = 'green' ) ) +
  geom_hline( yintercept = 0, linewidth = 0.5 ) +
  theme_bw( ) +
  theme( 
    aspect.ratio = 3/2,
    legend.position = 'none',
    panel.grid.major.y = element_line( color = 'gray', size = 0.5, linetype = 'dashed' ),
    panel.grid.minor.y = element_blank( ),
    axis.title = element_text( size = 15 ),
    panel.border = element_blank(),
    axis.line.y = element_line( color = 'black', linewidth = 0.5 ),
    axis.ticks.x = element_blank( ),
    axis.text.y = element_text( size = 5 ),
    plot.title = element_text( size = 15, hjust = 0.5 ) )
p2 <- ggplot( data = mydata_col, aes( x = Freq, y = start ) ) +
  geom_rect( aes( xmin = 0, xmax = Freq,
                  ymin = start, ymax = end ), fill = 'blue' ) +
  labs( x = 'Number of \n significant genes', y = NULL ) +
  scale_x_continuous( expand = c( 0.01, 0 ), 
                      breaks = c(0, 5, 10 ), labels = c(0, 5, 10) ) +
  scale_y_continuous( expand = c( 0.01, 0 ), breaks = c( chr_axis$end, -chr_axis$end ), labels = rep( 1:22, 2),
                      sec.axis = dup_axis( name = 'colon transverse                                 colon sigmoid') ) +
  geom_hline( yintercept = 0, linewidth = 0.5 ) +
  theme_bw( ) +
  theme( 
    aspect.ratio = 6,
    legend.position = 'none',
    panel.grid.major.y = element_line( color = 'gray', size = 0.5, linetype = 'dashed' ),
    panel.grid.minor.y = element_blank( ),
    # panel.grid.major.y = element_blank( ),
    # panel.grid.minor.y = element_blank( ),
    axis.title = element_text( size = 15 ),
    # panel.border = element_blank(),
    # axis.line.y = element_line( color = 'black', linewidth = 0.5 ),
    axis.ticks.y = element_blank( ),
    # axis.text.x = element_blank( ),
    axis.text.y = element_blank( ),
    plot.title = element_text( size = 15, hjust = 0.5 ) )
final_plot <- ( p1 | p2 )
final_plot
ggsave( '~/CY/MULTI/DA1.pdf', plot = final_plot, width = 6, height = 7, units = "in")



######################################## < Step 6 >  ########################################
keep_by_step <- function(x, k, include_first = TRUE) {
  stopifnot(is.numeric(x), length(k) == 1, is.numeric(k))
  n <- length(x)
  if (n == 0L) return(logical(0))
  
  keep <- rep(FALSE, n)

  i0 <- if (include_first) 1L else match(TRUE, !is.na(x))
  if (is.na(i0)) return(keep)
  
  keep[i0] <- TRUE
  last_val <- x[i0]
  
  if (i0 < n) {
    for (i in (i0 + 1L):n) {
      xi <- x[i]
      if (is.na(xi)) next
      if (xi - last_val >= k) {
        keep[i] <- TRUE
        last_val <- xi
      }
    }
  }
  keep
}
force2full <- function( x ){
  
  library( Matrix )
  is_full <- rankMatrix( x )[ 1 ] == nrow( x )
  
  if( is_full ){
    
    return( x )
    
  } else{
    
    x <- as.matrix( nearPD( x, corr = TRUE)$mat )
    return( x )
    
  }
  
}
setwd( '~/CY/Data' )
datapath_gtex <- 'GTEx/eQTL/'
datapath_gwas <- 'GWAS/Parkinson/'
parkinson <- fread( file = paste0( datapath_gwas, '332_PheCode.v1.0.fastGWA.gz') )
myfiles <- list.files( datapath_gtex )[ seq( 14, 38, 2 ) ]
brain_amygdala     <- fread( paste0( datapath_gtex, myfiles[ 1 ] ) )
brain_anterior     <- fread( paste0( datapath_gtex, myfiles[ 2 ] ) )
brain_caudate      <- fread( paste0( datapath_gtex, myfiles[ 3 ] ) )
brain_cerebellar   <- fread( paste0( datapath_gtex, myfiles[ 4 ] ) )
brain_cerebellum   <- fread( paste0( datapath_gtex, myfiles[ 5 ] ) )
brain_cortex       <- fread( paste0( datapath_gtex, myfiles[ 6 ] ) )
brain_frontal      <- fread( paste0( datapath_gtex, myfiles[ 7 ] ) )
brain_hippocampus  <- fread( paste0( datapath_gtex, myfiles[ 8 ] ) )
brain_hypothalamus <- fread( paste0( datapath_gtex, myfiles[ 9 ] ) )
brain_nucleus      <- fread( paste0( datapath_gtex, myfiles[ 10 ] ) )
brain_putamen      <- fread( paste0( datapath_gtex, myfiles[ 11 ] ) )
brain_spinal       <- fread( paste0( datapath_gtex, myfiles[ 12 ] ) )
brain_substantia   <- fread( paste0( datapath_gtex, myfiles[ 13 ] ) )
sample_Eur <- fread( 'ReferencePanel/integrated_call_samples_v3.20130502.ALL.panel',
                     header = T, fill = T ) %>% 
  as.data.frame( . ) %>% subset( ., super_pop == 'EUR' ) %>% .$sample
target_gene <- 'WDR6'  ######## chr3
target_chr  <- 3  ########
load( paste0( 'ReferencePanel/rs_chr', target_chr, '.RData' ) )
gds_path <-  paste0( 'ReferencePanel/ALL.chr',
                     target_chr,
                     '.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.gds')
############### Preprocess data (Parkinson) ###############
parkinson_small <- dplyr::select( parkinson, SNP, BETA, SE )
colnames( parkinson_small ) <- c( 'rs', 'by', 'byse' )

Tissues_TBD <- c( 'brain_amygdala', 'brain_caudate', 'brain_cerebellar', 
                  'brain_cerebellum', 'brain_cortex', 'brain_frontal',
                  'brain_hippocampus', 'brain_nucleus', 'brain_putamen', 
                  'brain_spinal', 'brain_substantia' )
############### Preprocess data (brain_amygdala) ###############
library( openxlsx )
wb <- createWorkbook( )
genofile <- snpgdsOpen( gds_path, readonly = TRUE )
mydata <- list( )
rs <- list( )
for( i in 1:length( Tissues_TBD ) ){
  # i <- 1
  data_int <- subset( get( Tissues_TBD[i] ), ID == target_gene & pval_nominal < 5e-6 & maf > 0.05 ) %>% 
               dplyr::select( 'CHR', 'ID', 'rs_ID', 'slope', 'slope_se', 'tss_distance' )
  colnames( data_int )   <- c( 'chr', 'gene', 'rs', 'bx', 'bxse', 'tss' )
  data_int$chr <- sub("^.*chr", "", data_int$chr )
  rs_int <- unique( data_int$rs ) %>% .[ . %in% parkinson_small$rs ] %>% .[ . %in% rs_all ]
  Tissue_int <- parkinson_small %>% filter( rs %in% rs_int ) %>%
                 merge( .,  data_int %>% filter( rs %in% rs_int ), by = 'rs' ) %>%
                  dplyr::select( ., rs, tss, bx, bxse, by, byse ) %>%
                   .[ order( .$tss ) ] %>%
                    subset( ., keep_by_step( .$tss, k = 50000 ) )
  addWorksheet( wb, Tissues_TBD[i] )
  writeData( wb, Tissues_TBD[i], Tissue_int )

  rs_int <- Tissue_int$rs
  rs[[ i ]] <- rs_int

  R_hat_int <- snpgdsGetGeno( genofile, sample.id = sample_Eur, snp.id = rs_int ) %>%
    cor( ) %>% force2full( )
  mydata[[ i ]] <- list( bx   = as.matrix( Tissue_int$bx ),
                         bxse = as.matrix( Tissue_int$bxse ),
                         by   = as.matrix( Tissue_int$by ),
                         byse = as.matrix( Tissue_int$byse ),
                         R_hat = R_hat_int )
}
closefn.gds( genofile )
saveWorkbook( wb, 'Brain.xlsx', overwrite = TRUE )
names( mydata ) <- paste0( 'Tissue', 1:length( mydata ) )
res <- MULTI( mydata, tissue_ref = 'Tissue1', r2 = 0 )

p <- c( )
for( i in 1:( length( res ) - 2 ) ){
  p[ i ] <- ifelse( res[[ i ]]$mu_beta > 0, 
                    pnorm( 0, res[[ i ]]$mu_beta, res[[ i ]]$se_beta, lower.tail = T ),
                    pnorm( 0, res[[ i ]]$mu_beta, res[[ i ]]$se_beta, lower.tail = F ) )
}
p <- as.numeric( sprintf( '%.3f', p ) )
names( p ) <- c( 'Amygdala', 'Caudate', 'Cerebellar', 'Cerebellum', 'Cortex', 
                 'Frontal', 'Hippocampus', 'Nucleus', 'Putamen', 'Spinal', 'Substantia' )
apply( HD_matrix, 1, function( x ) any( x > 0 & x < 0.5 ) )




##### Amygdala #####
Tissue_int <- data.frame(
  rs   = unlist( rs[ c( 1, 3, 10, 11 ) ] ),
  bx   = lapply( mydata[ c( 1, 3, 10, 11 ) ], function( x ) x$bx )   %>% unlist( ),
  bxse = lapply( mydata[ c( 1, 3, 10, 11 ) ], function( x ) x$bxse ) %>% unlist( ),
  by   = lapply( mydata[ c( 1, 3, 10, 11 ) ], function( x ) x$by )   %>% unlist( ),
  byse = lapply( mydata[ c( 1, 3, 10, 11 ) ], function( x ) x$byse ) %>% unlist( )
) %>% .[ ! duplicated( .$rs ), ]
genofile <- snpgdsOpen( gds_path, readonly = TRUE )
R_hat_int <- snpgdsGetGeno( genofile, sample.id = sample_Eur, snp.id = Tissue_int$rs ) %>%
  cor( ) %>% force2full( )
closefn.gds( genofile )
mydata_int <- list( bx   = as.matrix( Tissue_int$bx ),
                    bxse = as.matrix( Tissue_int$bxse ),
                    by   = as.matrix( Tissue_int$by ),
                    byse = as.matrix( Tissue_int$byse ),
                    R_hat = R_hat_int )
res_int <- MULTI_single( mydata_int, r2 = 0 )
ifelse( res_int$mu_beta > 0,
        pnorm( 0, res_int$mu_beta, res_int$se_beta, lower.tail = T ),
        pnorm( 0, res_int$mu_beta, res_int$se_beta, lower.tail = F ) )

##### Cerebellar #####
Tissue_int <- data.frame(
  rs   = unlist( rs[ c( 3, 1, 10, 11 ) ] ),
  bx   = lapply( mydata[ c( 3, 1, 10, 11 ) ], function( x ) x$bx )   %>% unlist( ),
  bxse = lapply( mydata[ c( 3, 1, 10, 11 ) ], function( x ) x$bxse ) %>% unlist( ),
  by   = lapply( mydata[ c( 3, 1, 10, 11 ) ], function( x ) x$by )   %>% unlist( ),
  byse = lapply( mydata[ c( 3, 1, 10, 11 ) ], function( x ) x$byse ) %>% unlist( )
) %>% .[ ! duplicated( .$rs ), ]
genofile <- snpgdsOpen( gds_path, readonly = TRUE )
R_hat_int <- snpgdsGetGeno( genofile, sample.id = sample_Eur, snp.id = Tissue_int$rs ) %>%
  cor( ) %>% force2full( )
closefn.gds( genofile )
mydata_int <- list( bx   = as.matrix( Tissue_int$bx ),
                    bxse = as.matrix( Tissue_int$bxse ),
                    by   = as.matrix( Tissue_int$by ),
                    byse = as.matrix( Tissue_int$byse ),
                    R_hat = R_hat_int )
res_int <- MULTI_single( mydata_int, r2 = 0 )
ifelse( res_int$mu_beta > 0, 
        pnorm( 0, res_int$mu_beta, res_int$se_beta , lower.tail = T ),
        pnorm( 0, res_int$mu_beta, res_int$se_beta , lower.tail = F ) )

##### Frontal #####
Tissue_int <- data.frame(
  rs   = unlist( rs[ c( 6, 7 ) ] ),
  bx   = lapply( mydata[ c( 6, 7 ) ], function( x ) x$bx )   %>% unlist( ),
  bxse = lapply( mydata[ c( 6, 7 ) ], function( x ) x$bxse ) %>% unlist( ),
  by   = lapply( mydata[ c( 6, 7 ) ], function( x ) x$by )   %>% unlist( ),
  byse = lapply( mydata[ c( 6, 7 ) ], function( x ) x$byse ) %>% unlist( )
) %>% .[ ! duplicated( .$rs ), ]
genofile <- snpgdsOpen( gds_path, readonly = TRUE )
R_hat_int <- snpgdsGetGeno( genofile, sample.id = sample_Eur, snp.id = Tissue_int$rs ) %>%
  cor( ) %>% force2full( )
closefn.gds( genofile )
mydata_int <- list( bx   = as.matrix( Tissue_int$bx ),
                    bxse = as.matrix( Tissue_int$bxse ),
                    by   = as.matrix( Tissue_int$by ),
                    byse = as.matrix( Tissue_int$byse ),
                    R_hat = R_hat_int )
res_int <- MULTI_single( mydata_int, r2 = 0 )
ifelse( res_int$mu_beta > 0,
        pnorm( 0, res_int$mu_beta, res_int$se_beta , lower.tail = T ),
        pnorm( 0, res_int$mu_beta, res_int$se_beta , lower.tail = F ) )

##### Hippocampus #####
Tissue_int <- data.frame(
  rs   = unlist( rs[ c( 7, 6 ) ] ),
  bx   = lapply( mydata[ c( 7, 6 ) ], function( x ) x$bx )   %>% unlist( ),
  bxse = lapply( mydata[ c( 7, 6 ) ], function( x ) x$bxse ) %>% unlist( ),
  by   = lapply( mydata[ c( 7, 6 ) ], function( x ) x$by )   %>% unlist( ),
  byse = lapply( mydata[ c( 7, 6 ) ], function( x ) x$byse ) %>% unlist( )
) %>% .[ ! duplicated( .$rs ), ]
genofile <- snpgdsOpen( gds_path, readonly = TRUE )
R_hat_int <- snpgdsGetGeno( genofile, sample.id = sample_Eur, snp.id = Tissue_int$rs ) %>%
  cor( ) %>% force2full( )
closefn.gds( genofile )
mydata_int <- list( bx   = as.matrix( Tissue_int$bx ),
                    bxse = as.matrix( Tissue_int$bxse ),
                    by   = as.matrix( Tissue_int$by ),
                    byse = as.matrix( Tissue_int$byse ),
                    R_hat = R_hat_int )
res_int <- MULTI_single( mydata_int, r2 = 0 )
ifelse( res_int$mu_beta > 0,
        pnorm( 0, res_int$mu_beta, res_int$se_beta , lower.tail = T ),
        pnorm( 0, res_int$mu_beta, res_int$se_beta , lower.tail = F ) )

##### Spinal #####
Tissue_int <- data.frame(
  rs   = unlist( rs[ c( 10, 1, 3, 11 ) ] ),
  bx   = lapply( mydata[ c( 10, 1, 3, 11) ], function( x ) x$bx )   %>% unlist( ),
  bxse = lapply( mydata[ c( 10, 1, 3, 11 ) ], function( x ) x$bxse ) %>% unlist( ),
  by   = lapply( mydata[ c( 10, 1, 3, 11 ) ], function( x ) x$by )   %>% unlist( ),
  byse = lapply( mydata[ c( 10, 1, 3, 11 ) ], function( x ) x$byse ) %>% unlist( )
) %>% .[ ! duplicated( .$rs ), ]
genofile <- snpgdsOpen( gds_path, readonly = TRUE )
R_hat_int <- snpgdsGetGeno( genofile, sample.id = sample_Eur, snp.id = Tissue_int$rs ) %>%
  cor( ) %>% force2full( )
closefn.gds( genofile )
mydata_int <- list( bx   = as.matrix( Tissue_int$bx ),
                    bxse = as.matrix( Tissue_int$bxse ),
                    by   = as.matrix( Tissue_int$by ),
                    byse = as.matrix( Tissue_int$byse ),
                    R_hat = R_hat_int )
res_int <- MULTI_single( mydata_int, r2 = 0 )
ifelse( res_int$mu_beta > 0,
        pnorm( 0, res_int$mu_beta, res_int$se_beta , lower.tail = T ),
        pnorm( 0, res_int$mu_beta, res_int$se_beta , lower.tail = F ) )

##### Substantia #####
Tissue_int <- data.frame(
  rs   = unlist( rs[ c( 11, 1, 3, 10 ) ] ),
  bx   = lapply( mydata[ c( 11, 1, 3, 10 ) ], function( x ) x$bx )   %>% unlist( ),
  bxse = lapply( mydata[ c( 11, 1, 3, 10 ) ], function( x ) x$bxse ) %>% unlist( ),
  by   = lapply( mydata[ c( 11, 1, 3, 10 ) ], function( x ) x$by )   %>% unlist( ),
  byse = lapply( mydata[ c( 11, 1, 3, 10 ) ], function( x ) x$byse ) %>% unlist( )
) %>% .[ ! duplicated( .$rs ), ]
genofile <- snpgdsOpen( gds_path, readonly = TRUE )
R_hat_int <- snpgdsGetGeno( genofile, sample.id = sample_Eur, snp.id = Tissue_int$rs ) %>%
  cor( ) %>% force2full( )
closefn.gds( genofile )
mydata_int <- list( bx   = as.matrix( Tissue_int$bx ),
                    bxse = as.matrix( Tissue_int$bxse ),
                    by   = as.matrix( Tissue_int$by ),
                    byse = as.matrix( Tissue_int$byse ),
                    R_hat = R_hat_int )
res_int <- MULTI_single( mydata_int, r2 = 0 )
ifelse( res_int$mu_beta > 0,
        pnorm( 0, res_int$mu_beta, res_int$se_beta , lower.tail = T ),
        pnorm( 0, res_int$mu_beta, res_int$se_beta , lower.tail = F ) )


##### No integration #####
HD_matrix <- 1 - res$`Hellinger Distance Matrix`
colnames( HD_matrix ) <- c( 'Amygdala', 'Caudate', 'Cerebellar', 'Cerebellum', 'Cortex', 'Frontal',
                            'Hippocampus', 'Nucleus', 'Putamen', 'Spinal', 'Substantia' )
rownames( HD_matrix ) <- c( 'Amygdala', 'Caudate', 'Cerebellar', 'Cerebellum', 'Cortex', 'Frontal',
                            'Hippocampus', 'Nucleus', 'Putamen', 'Spinal', 'Substantia' )
df <- melt( HD_matrix, varnames = c( 'row', 'col' ), value.name = 'HD' )
df$ri <- as.integer( df$row )
df$ci <- as.integer( df$col )
df_half <- subset( df, ri >= ci )
df_diag <- subset( df_half, ri == ci)
df_diag$label <- c( '- \n < 0.001', '+ \n < 0.001', '+ \n < 0.001', '- \n < 0.001', '- \n 0.146', '+ \n 0.445', '- \n 0.071',    '- \n 0.479', '- \n 0.002', '- \n 0.301', '- \n < 0.001'  )
p1 <- ggplot( df_half, aes( x = col, y = row, fill = HD ) ) +
  geom_tile() +
  # labs( title = 'Causal results for different' ) +
  ggtitle( 'Causal inference results with MULTI across brain regions' ) +
  scale_fill_gradientn(
    colours = c( '#eff3ff',
                 '#6baed6', '#fee0d2',  # 蓝系（0 → 0.5）
                 '#fb6a4a', '#a50f15'),  # 红系（0.5→1）
    values  = scales::rescale(c(0, 0.5, 0.5, 1)),  # 0.5 重复 => 断点
    limits  = c( 0, 1 ), na.value = 'white' ) +
  coord_fixed( ) +
  # geom_text( data = df_diag, aes( label = label ), color = 'black', size = 3, fontface = 'bold' ) +
  geom_text( data = df_diag, aes( label = label ), size = 3, fontface = 'bold', colour = 'white' ) +
  scale_y_discrete( limits = rev ) +
  theme_minimal( base_size = 12 ) +
  theme( panel.grid = element_blank(),
         plot.title = element_text( hjust = 0.5 ),
         axis.title = element_blank(),
         axis.text.x = element_text( angle = 45, hjust = 1, size = 12 ),
         axis.text.y = element_text( hjust = 1, size = 12 ) )
p1
# ggsave( '~/CY/MULTI/DA2.pdf', plot = p1, width = 6, height = 6, units = "in")

