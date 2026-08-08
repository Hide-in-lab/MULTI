# MULTI
**MULTI** (**M**ulti-tissue **U**nified **L**ikelihood-based **T**ranscriptomic **I**ntegration) is a multi-tissue Mendelian randomization method designed to identify tissue-specific causal genes underlying complex traits.

Assuming that we have _**K**_ tissues, one of which is designated as the target tissue ($T_1$) and the remaining tissues are considered supportive tissues, MULTI leverages information from supportive tissues to enhance the estimation of causal effects between candidate genes and diseases in the target tissue, thereby reducing type I error rates and improving statistical power.

# RARE
**RARE** is a multivariable Mendelian randomization method, and is the short name of 'MVMR incorporating **R**are variants **A**ccounting for multiple **R**isk factors and shared horizontal pl**E**iotopy'.

RARE is the only method (2024-07-31) that accounts for the impact of rare variants in causal inference while simultaneously considers UHP and CHP.

![Image text](https://github.com/Hide-in-lab/RARE/blob/main/image/Github_RARE.jpg)


install.packages( 'devtools' )  
library( devtools )  
install_github( 'Hide-in-lab/MULTI@main', force  = T )
