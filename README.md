# MULTI
**MULTI** (**M**ulti-tissue **U**nified **L**ikelihood-based **T**ranscriptomic **I**ntegration) is a multi-tissue Mendelian randomization method designed to identify tissue-specific causal genes underlying complex traits.

Assuming that we have _**K**_ tissues, one of which is designated as the target tissue (_**T**_**<sub>1</sub>**) and the remaining tissues are considered supportive tissues (***T<sub>i</sub>***, $i \neq 0$), MULTI leverages information from supportive tissues to enhance the estimation of causal effects between gene expression level and disease in the target tissue, thereby reducing type I error rates and improving statistical power.



RARE is the only method (2024-07-31) that accounts for the impact of rare variants in causal inference while simultaneously considers UHP and CHP.

![Image text](https://github.com/Hide-in-lab/RARE/blob/main/image/Github_RARE.jpg)


install.packages( 'devtools' )  
library( devtools )  
install_github( 'Hide-in-lab/MULTI@main', force  = T )
