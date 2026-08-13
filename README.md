<div align="justify">


# MULTI
**MULTI** (**M**ulti-tissue **U**nified **L**ikelihood-based **T**ranscriptomic **I**ntegration) is a multi-tissue Mendelian randomization method designed to identify tissue-specific causal genes underlying complex traits.
Assuming that we have _**K**_ tissues, one of which is designated as the target tissue (_**T**_**<sub>1</sub>**) and the remaining tissues are considered supportive tissues (_**T<sub>k</sub>**_, k = 2, 3, ..., K), MULTI leverages information from supportive tissues to enhance the estimation of causal effects between gene expression level and disease in the target tissue, thereby reducing type I error rates and improving statistical power.

<p align="center">
  <img src="https://github.com/Hide-in-lab/MULTI/blob/SupplementaryResults/Github/Figure%201.jpg?raw=true" width="50%" />
  <br>
  <b>Fig.1 Graphic abstract of MULTI</b>
</p>

# Data Input
Assuming the k-th (_k_ = 1, 2, ..., _K_) tissue has _**p<sub>k</sub>**_ eligible IVs (typically cis-eQTL). And as shown in **Fig. 1A** of the paper, MULTI requires three types of input data for each tissue:

**1)** _**Estimated IV-to-exposure effect size vector**_ $`\hat{\boldsymbol{\gamma}}_k`$ and its _**corresponding diagonal standard error matrix**_ $`\hat{\mathbf{S}}_{\gamma_k}`$ (_e.g._, GTEx Portal), where

$$ \hat{\boldsymbol{\gamma}}_k = (\hat{\gamma}_{k1},\hat{\gamma}_{k2},\cdots,\hat{\gamma}_{kp_k})^{\prime},
\quad
\hat{\boldsymbol{S}}_{\gamma_k} = \mathrm{diag}(\hat{S}_{\gamma_{k1}},\hat{S}_{\gamma_{k2}},\cdots,\hat{S}_{\gamma_{kp_k}}) $$

**2)** _**Estimated IV-to-outcome effect size vector**_ $`\hat{\boldsymbol{\Gamma}}_k`$ and its _**corresonding diagonal standard error matrix**_ $`\hat{\mathbf{S}}_{\Gamma_k}`$ (_e.g._, GWAS Catalog), where

$$ \hat{\boldsymbol{\Gamma}}_k = (\hat{\Gamma}_{k1},\hat{\Gamma}_{k2},\cdots,\hat{\Gamma}_{kp_k})^{\prime},
\quad
\hat{\boldsymbol{S}}_{\Gamma_k} = \mathrm{diag}(\hat{S}_{\Gamma_{k1}},\hat{S}_{\Gamma_{k2}},\cdots,\hat{S}_{\Gamma_{kp_k}}) $$

**3)** _**Pairwise correlation matrix**_ $`\hat{\boldsymbol{R}}_k`$ representing the linkage disequilibrium (LD) among the elegible IVs. This matrix can be estimated using an independet reference panel database, e.g. _**1000 Genomes Project Phase 3**_.

# Workflow
As illustrated in **Fig. 1B and 1C**, the workflow of MULTI comprises four primary steps:

**1)** Using **Variantional Expectation-Maximum algorithm (VEM)** to independently calculate the **tissue-specific** gene-outcome effects $`\hat{\beta_k}`$ and their standard errors $`\hat{S}_{\beta_k1}`$, _k_ = 1, 2, ..., _K_. 

**2)** Computing the Hellinger distance (HD) between all supportive tissues and the target tissue to represent the **tissue similarity**.

**3)** For the k-th (_k_ = 2, ..., _K_) tissue, if the $\color{red}{HD \leq \text{user-specified cutoff}}$, the eligible IVs of this tissue are incorporated into the IV pool of the target tissue.

**4)** Re-estimate the final gene-outcome causal effect $`\hat{\beta}`$ using the expanded IV pool in the target tissue.

**Note:** <br>
In the manuscript, we provided three reference similarity thresholds (**0.1**, **0.3**, and **0.5**). Simulation results demonstrated that, as the similarity threshold relaxed, the type I error rate of MULTI increased slightly; however, it remained well controlled below the nominal level of **0.05** in most scenarios. Meanwhile, this resulted in an approximately **10% ~ 15%** improvement in statistical power.

In practical applications, researchers who aim to strictly control false-positive discoveries may choose a lower similarity threshold. Conversely, researchers who prioritize the identification of as many potential causal genes as possible may adopt a higher similarity threshold. Based on the simulation results presented in this study, we recommend using **cutoff = 0.5** as a reference threshold for practical applications. Nevertheless, researchers can adjust the similarity threshold according to their specific research objectives and desired balance between false-positive control and discovery power.


# Installation
Install this tool by use of the 'devtools' package. Note that MULTI partly depends on the C++ languange, thus you should appropriately set Rtools and X code for Windows, Mac OS/X, and Linux, respectively.
```
install.packages( 'devtools' )  
library( devtools )  
install_github( 'Hide-in-lab/MULTI@main', force  = T )
```

# Usage
**Step 1)** $\color{red}{\textbf{Preparing Input Data}}$

Users should prepare the `bx`, `bxse`, `by`, `byse`, and the estimated LD matrix `R_hat` for each tissue. Users must $\color{red}{\textbf{strictly format these variables}}$ as the example code to ensure the `MULTI` package runs correctly.

```
##### Example Code #####
mydata <- list( )
for( i in 1:length( Tissues_TBD ) ){

  mydata[[ i ]] <- list( bx   = as.matrix( Tissue_int$bx ),
                         bxse = as.matrix( Tissue_int$bxse ),
                         by   = as.matrix( Tissue_int$by ),
                         byse = as.matrix( Tissue_int$byse ),
                         R_hat = R_hat_int )

}
names( mydata ) <- paste0( 'Tissue', 1:length( mydata ) )
```
Then you will get a data list like this:
<p align="center">
  <img src="https://github.com/Hide-in-lab/MULTI/blob/SupplementaryResults/Github/DataInput.png?raw=true" width="50%" />
</p>

**Step 2)** $\color{red}{\textbf{Calculating the tissue similarity}}$

Call the `MULTI()` function to calculate the tissue similarity matrix.
```
library( MULTI )
res <- MULTI( DataList = mydata, tissue_ref = "Tissue1", r2 = 0.01, cut_off = 0.5, iter_times = 500, ELBO_tol = 1e-6 )
res

##### Parameters Explanation #####
### DataList: Data input
### tissue_ref: To specify the target tissue. Default is `Tissue 1`.
### r2: To simplify the LD matrix. If the LD value between SNPs is below this specified threshold, it is forced to 0. r2 ranges from 0 to 1, and default is 0.01.
### cut_off: To specify the similarity cutoff. If the Hellinger distance between the supportive tissue and the target tissue, this supportive tissue will be regarded as the similar tissue to be integrated. cut_off ranges from 0 to 1, and default is 0.5.
### iter_times: To specify the iteration times. MULTI utilizes the VEM algorithm and typically converges within a few dozen iterations. Default is 500.
### ELBO_tol: To specify the convergence tolerance of the model. Generally, a smaller value yields more accurate results; however, setting it too small may result in excessively long computation times. Default is 1e-6.
```
Then you will get a data list like this:
<p align="center">
  <img src="https://github.com/Hide-in-lab/MULTI/blob/SupplementaryResults/Github/HD.png?raw=true" width="90%" />
</p>

**Step 3)** $\color{red}{\textbf{Calculating the causal effect}}$

Finally, users should integrate the SNP information of recommended supportive tissue(s) to the target tissue, and create another `DataList` like **Step 2**. Then call the `MULTI_single()` function to calculate the final gene-outcome causal effect.

Note that, users should $\color{red}{\textbf{re-estimate the LD matrix using the integrated IV pool}}$.
```
##### Example Code #####
Tissue_int <- data.frame(
  rs   = unlist( rs[ c( 1, 3, 10, 11 ) ] ),
  bx   = lapply( mydata[ c( 1, 3, 10, 11 ) ], function( x ) x$bx )   %>% unlist( ),
  bxse = lapply( mydata[ c( 1, 3, 10, 11 ) ], function( x ) x$bxse ) %>% unlist( ),
  by   = lapply( mydata[ c( 1, 3, 10, 11 ) ], function( x ) x$by )   %>% unlist( ),
  byse = lapply( mydata[ c( 1, 3, 10, 11 ) ], function( x ) x$byse ) %>% unlist( )
)
mydata_int <- list( bx   = as.matrix( Tissue_int$bx ),
                    bxse = as.matrix( Tissue_int$bxse ),
                    by   = as.matrix( Tissue_int$by ),
                    byse = as.matrix( Tissue_int$byse ),
                    R_hat = R_hat_int )  ## R_hat_int: re-estimated LD matrix
res_int <- MULTI_single( mydata_int, r2 = 0.01, ter_times = 500, ELBO_tol = 1e-6 )
p_value <- ifelse( res_int$mu_beta > 0,
        pnorm( 0, res_int$mu_beta, res_int$se_beta, lower.tail = T ),
        pnorm( 0, res_int$mu_beta, res_int$se_beta, lower.tail = F ) )
p_value
```
Then you will get a data list like this:
<p align="center">
  <img src="https://github.com/Hide-in-lab/MULTI/blob/SupplementaryResults/Github/Final.png?raw=true" width="30%" />
</p>


# Reference

$\color{red}{\textbf{Please kindly cite the following paper if you use the}}$  `MULTI` $\color{red}{\textbf{package:}}$

Yu Cheng<sup>+</sup>, Shuhan Liu<sup>+</sup>, Xinjia Ruan<sup>+</sup>, Zhonghua Li, Liyun Jiang<sup> #</sup>, Tiantian Liu<sup> #</sup>, Fangrong Yan<sup> #</sup>, **Multi-tissue integrated Mendelian randomization method identifies disease risk genes**, Briefings in Bioinformatics, Volume 27, Issue 4, July 2026, bbag414, https://doi.org/10.1093/bib/bbag414

# Development

This package is developed and maintained by **Yu Cheng**. 

$\color{red}{\textbf{To avoid duplicating effort in preparing the data required for calculating linkage disequilibrium (LD) matrices}}$, researchers may request the preprocessed datasets directly from the first author via email.

**Contact e-mail**: yucheng.cpu@foxmail.com


</div>



