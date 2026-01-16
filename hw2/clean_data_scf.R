# From: https://asdfree.com/survey-of-consumer-finances-scf.html

library(haven)
library(stringr)
library(survey)
library(mitools)

scf_MIcombine <-
  function (results, variances, call = sys.call(), df.complete = Inf, ...) {
    m <- length(results)
    oldcall <- attr(results, "call")
    if (missing(variances)) {
      variances <- suppressWarnings(lapply(results, vcov))
      results <- lapply(results, coef)
    }
    vbar <- variances[[1]]
    cbar <- results[[1]]
    for (i in 2:m) {
      cbar <- cbar + results[[i]]
      # MODIFICATION:
      # vbar <- vbar + variances[[i]]
    }
    cbar <- cbar/m
    # MODIFICATION:
    # vbar <- vbar/m
    evar <- var(do.call("rbind", results))
    r <- (1 + 1/m) * evar/vbar
    df <- (m - 1) * (1 + 1/r)^2
    if (is.matrix(df)) df <- diag(df)
    if (is.finite(df.complete)) {
      dfobs <- ((df.complete + 1)/(df.complete + 3)) * df.complete *
        vbar/(vbar + evar)
      if (is.matrix(dfobs)) dfobs <- diag(dfobs)
      df <- 1/(1/dfobs + 1/df)
    }
    if (is.matrix(r)) r <- diag(r)
    rval <- list(coefficients = cbar, variance = vbar + evar *
                   (m + 1)/m, call = c(oldcall, call), nimp = m, df = df,
                 missinfo = (r + 2/(df + 3))/(r + 1))
    class(rval) <- "MIresult"
    rval
  }

scf_dta_import <-
  function( this_url ){
    
    this_tf <- tempfile()
    
    download.file( this_url , this_tf , mode = 'wb' )
    
    this_tbl <- read_dta( this_tf )
    
    this_df <- data.frame( this_tbl )
    
    file.remove( this_tf )
    
    names( this_df ) <- tolower( names( this_df ) )
    
    this_df
  }

# Download data

scf_df <- scf_dta_import( "https://www.federalreserve.gov/econres/files/scf2022s.zip" )

ext_df <- scf_dta_import( "https://www.federalreserve.gov/econres/files/scfp2022s.zip" )

scf_rw_df <- scf_dta_import( "https://www.federalreserve.gov/econres/files/scf2022rw1s.zip" )

stopifnot( nrow( scf_df ) == nrow( scf_rw_df ) * 5 )
stopifnot( nrow( scf_df ) == nrow( ext_df ) )

stopifnot( all( sort( intersect( names( scf_df ) , names( ext_df ) ) ) == c( 'y1' , 'yy1' ) ) )
stopifnot( all( sort( intersect( names( scf_df ) , names( scf_rw_df ) ) ) == c( 'y1' , 'yy1' ) ) )
stopifnot( all( sort( intersect( names( ext_df ) , names( scf_rw_df ) ) ) == c( 'y1' , 'yy1' ) ) )

scf_rw_df[ , 'y1' ] <- NULL

scf_df[ , 'five' ] <- 5

s1_df <- scf_df[ str_sub( scf_df[ , 'y1' ] , -1 , -1 ) == 1 , ]
s2_df <- scf_df[ str_sub( scf_df[ , 'y1' ] , -1 , -1 ) == 2 , ]
s3_df <- scf_df[ str_sub( scf_df[ , 'y1' ] , -1 , -1 ) == 3 , ]
s4_df <- scf_df[ str_sub( scf_df[ , 'y1' ] , -1 , -1 ) == 4 , ]
s5_df <- scf_df[ str_sub( scf_df[ , 'y1' ] , -1 , -1 ) == 5 , ]

scf_imp <- list( s1_df , s2_df , s3_df , s4_df , s5_df )

scf_list <- lapply( scf_imp , merge , ext_df )

scf_rw_df[ is.na( scf_rw_df ) ] <- 0

scf_rw_df[ , paste0( 'wgt' , 1:999 ) ] <-
scf_rw_df[ , paste0( 'wt1b' , 1:999 ) ] * scf_rw_df[ , paste0( 'mm' , 1:999 ) ]

scf_rw_df <- scf_rw_df[ , c( 'yy1' , paste0( 'wgt' , 1:999 ) ) ]

scf_list <- lapply( scf_list , function( w ) w[ order( w[ , 'yy1' ] ) , ] )

scf_rw_df <- scf_rw_df[ order( scf_rw_df[ , 'yy1' ] ) , ]

scf_design <- 
  svrepdesign( 
    weights = ~wgt , 
    repweights = scf_rw_df[ , -1 ] , 
    data = imputationList( scf_list ) , 
    scale = 1 ,
    rscales = rep( 1 / 998 , 999 ) ,
    mse = FALSE ,
    type = "other" ,
    combined.weights = TRUE
  )

scf_design <- 
  update( 
    scf_design , 
    
    hhsex = factor( hhsex , levels = 1:2 , labels = c( "male" , "female" ) ) ,
    
    married = as.numeric( married == 1 ) ,
    
    edcl = 
      factor( 
        edcl , 
        levels = 1:4 ,
        labels = 
          c( 
            "less than high school" , 
            "high school or GED" , 
            "some college" , 
            "college degree" 
          ) 
      )
    
  )

# All of this to do this: assets first

age_num <- scf_MIcombine( with( scf_design , svyby( ~ five , ~ age , unwtd.count ) ) )

networth_per_age <- scf_MIcombine( with( scf_design ,
                     svyby( ~ networth , ~age  , svymean )
) )

networth_per_age_median <- scf_MIcombine( with( scf_design ,
                                                svyby( ~ networth , ~age  , svyquantile , 0.5)
) )

vals_networth <- networth_per_age_median$coefficients

# cut at 21 - 85 yo

min_age <- which(names(vals_networth) == 21)
max_age <- which(names(vals_networth) == 85)

leftover <- mean(vals_networth[max_age:length(vals_networth)])
vals_networth <- vals_networth[min_age:max_age]

n_blocks <- 5
length_data <- length(vals_networth)
block_size <- length_data/n_blocks
blocks_beg <- seq(1,65,by=block_size)
blocks_end <- blocks_beg + block_size-1

avg_median_assets <- rep(NA,n_blocks+1)

for(i in 1:n_blocks){
  b <- blocks_beg[i]
  en <- blocks_end[i]
  
  avg_median_assets[i] <- mean(vals_networth[b:en])
}

avg_median_assets[n_blocks+1] <- leftover

plot(avg_median_assets,type = "l")

## Same exercise, but for income

income_per_age_median <- scf_MIcombine( with( scf_design ,
                                                svyby( ~ income , ~age  , svyquantile , 0.5)
) )

vals_income <- income_per_age_median$coefficients

plot(vals_income,type = "l")

# cut at 21 - 85 yo

min_age <- which(names(vals_income) == 21)
max_age <- which(names(vals_income) == 85)

vals_income <- vals_income[min_age:max_age]

n_blocks <- 5
length_data <- length(vals_income)
block_size <- length_data/n_blocks
blocks_beg <- seq(1,65,by=block_size)
blocks_end <- blocks_beg + block_size-1

avg_median_income <- rep(NA,n_blocks)

for(i in 1:n_blocks){
  b <- blocks_beg[i]
  en <- blocks_end[i]
  
  avg_median_income[i] <- mean(vals_income[b:en])
}

plot(avg_median_income, type = "l")

# align vectors

avg_median_income <- c(avg_median_income,NA)

df_sum <- data.frame(assets = avg_median_assets,inc = avg_median_income)

write.csv(df_sum,file = "~/Documents/GitHub/ECON890/hw2/summary_stats.csv")
