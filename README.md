# MULTI
**MULTI** (**M**ulti-tissue **U**nified **L**ikelihood-based **T**ranscriptomic **I**ntegration) is a multi-tissue Mendelian randomization method designed to identify tissue-specific causal genes underlying complex traits. \
Assuming that we have _**K**_ tissues, one of which is designated as the target tissue (_**T**_**<sub>1</sub>**) and the remaining tissues are considered supportive tissues (_**T<sub>i</sub>**_, $i \neq 0$), MULTI leverages information from supportive tissues to enhance the estimation of causal effects between gene expression level and disease in the target tissue, thereby reducing type I error rates and improving statistical power.

# Data Input
As shown in Fig. 1 of the paper, MULTI requires three types of input data: \
1> The estimated IV-to-exposure effect size vector and its corresonding diagonal standard error matrix \
$$ \boldsymbol{\hat{\gamma}}_k = (\hat{\gamma}_{k1},\hat{\gamma}_{k2},\cdots,\hat{\gamma}_{kp_k})^{\prime} $$,
$$ \boldsymbol{\hat{S}}_k = \mathrm{diag}(\hat{S}_{\gamma_{k1}},\hat{S}_{\gamma_{k2}},\cdots,\hat{S}_{\gamma_{kp_k}}) $$




![Image text](https://github.com/Hide-in-lab/MULTI/blob/SupplementaryResults/Figure%201.jpg)


install.packages( 'devtools' )  
library( devtools )  
install_github( 'Hide-in-lab/MULTI@main', force  = T )
