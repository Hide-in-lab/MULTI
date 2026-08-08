<div align="justify">


# MULTI
**MULTI** (**M**ulti-tissue **U**nified **L**ikelihood-based **T**ranscriptomic **I**ntegration) is a multi-tissue Mendelian randomization method designed to identify tissue-specific causal genes underlying complex traits.
Assuming that we have _**K**_ tissues, one of which is designated as the target tissue (_**T**_**<sub>1</sub>**) and the remaining tissues are considered supportive tissues (_**T<sub>k</sub>**_, k = 2, 3, ..., K), MULTI leverages information from supportive tissues to enhance the estimation of causal effects between gene expression level and disease in the target tissue, thereby reducing type I error rates and improving statistical power.

# Data Input
Assuming the k-th (_k_ = 1, 2, ..., _K_) tissue has _**p<sub>k</sub>**_ eligible IVs (typically cis-eQTL). And as shown in **Fig. 1A** of the paper, MULTI requires three types of input data for each tissue:

**1)** _**Estimated IV-to-exposure effect size vector**_ $`\hat{\boldsymbol{\gamma}}_k`$ and its _**corresponding diagonal standard error matrix**_ $`\hat{\mathbf{S}}_{\Gamma_k}`$ (_e.g._, GTEx Portal), where

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

<p align="center">
  <img src="https://github.com/Hide-in-lab/MULTI/blob/SupplementaryResults/Figure%201.jpg?raw=true" width="50%" />
  <br>
  <b>Fig.1 Workflow of MULTI</b>
</p>

# Installation
Install this tool by use of the 'devtools' package. Note that MULTI partly depends on the C++ languange, thus you should appropriately set Rtools and X code for Windows, Mac OS/X, and Linux, respectively.
```
install.packages( 'devtools' )  
library( devtools )  
install_github( 'Hide-in-lab/MULTI@main', force  = T )
```

# Usage
When using the MULTI package, users should 

**1)** first call the `MULTI()` function to calculate the tissue-specific effects. This function identifies which supportive tissues are similar to the target tissue, guiding users on which tissues to integrate;
```
MULTI( DataList, tissue_ref = "Tissue1", r2 = 0.01, cut_off = 0.5, iter_times = 500, ELBO_tol = 1e-6 )
```
**2)** then call the `MULTI_single()` function to calculate the final gene-outcome causal effect. Note that, unlike the `MULTI()` function, the input for `MULTI_single()` is the newly integrated IV pool.
```
MULTI_single( DataList, double r2 = 0.01, iter_times = 500, ELBO_tol = 1e-6 )
```
Generally, users only need to adjust the data input; the remaining parameters in the function can be left at their default values.

# Reference
Yu Cheng<sup>+</sup>, Shuhan Liu<sup>+</sup>, Xinjia Ruan<sup>+</sup>, Zhonghua Li, Liyun Jiang<sup> #</sup>, Tiantian Liu<sup> #</sup>, Fangrong Yan<sup> #</sup>, **Multi-tissue integrated Mendelian randomization method identifies disease risk genes**, Briefings in Bioinformatics, Volume 27, Issue 4, July 2026, bbag414, https://doi.org/10.1093/bib/bbag414

**Contact e-mail**: yucheng.cpu@foxmail.com


</div>



