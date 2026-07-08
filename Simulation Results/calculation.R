setwd( '/Volumes/Hide in lab/Yu Cheng/博士课题/Multi-tissue MR/Results/Simulation/n_X/' )
files <- list.files( pattern = '^result_1_1_\\d+_nX.RData$' )
results <- c( )
for( i in 1:10 ){
  # i <- 1
  # env <- new.env( )
  load( files[i] )
  results_int <- unlist( lapply( sim_res, function( x ) x$typeIerror ) )
  results <- c( results, results_int )
}
result_typeI <- table( names( results ), results )
result_typeI/500



library(dplyr)
library(tidyr)
results <- c( )
for( i in 1:5 ){
  # i <- 1
  load( files[i] )
  results_int <- unlist( lapply( sim_res, function( x ) x$bias ) )
  results <- c( results, results_int )
}
