# Study-V004-Statistical-Model-Collection


### 1> LinearRegression
LinearRegression is perhaps the simplest way to relate a continuous response variable to multiple explanatory variables. This may arise from observing several variables together and investigating which variables correlate with the response variable. Or it could arise from conducting an experiment, where we carefully assign values of explanatory variables to randomly selected subjects and try to establish a cause and effect relationship. 
<img src="https://user-images.githubusercontent.com/31917400/48806376-042c3380-ed12-11e8-8f37-67ef2e4e9ce7.jpg" />

 - The common choice for prior on the σ2 would be an `InverseGamma`
 - The common choice for prior on the β would be a `Normal`. Or we can do a multivariate normal for all of the β at once. This is conditionally conjugate and allows us to do **Gibbs-Sampling**.
 - Another common choice for prior on the β would be a `Double Exponential`(Laplace Distribution):`P(β) = 1/2*exp(-|β|)`. It has a sharp peak at `β=0`, which can be useful for **variable selection** among our X's because it'll favor values in your 0 for these βssss. This is related to the popular regression technique known as the `LASSO` ShrinkRegression(for more interpretable model with less variables). 
 - The interpretation of the β coefficients: While holding all other X variables constant, if `x1` increases by one, then the `mean of y` is expected to increase by **β1**. That is, **β1** describes how the `mean of y` changes with changes in `x1`, while **accounting for all the other X variables**. 
 - Assumptions: 
   - Given each X, `y1, y2,..` are **independent** of each other
   - `y1, y2,..` share the same variance.
   - E[**ε**] = 0
   - var(**ε**) = σ2
   - cov(**ε**, y) = 0
   - If they are not met, Hierachical model can address it. 





### 2> ANOVA
ANOVA is used when we have **categorical explanatory variables** so that the observations`Y` belong to groups. In ANOVA, we compare the variability of responses(Y) `within groups` to the variability of responses(Y) `between groups`. If the variability **between groups** is large, relative to the variability within the groups, we conclude that there is a `grouping effect`. Factors can have `2 categories` or `many levels`. For example, low and high or true and false. Or they can have many levels. 
 - Let's say we are going to conduct an online marketing experiment.
   - we might experiment with two factors of website design - sound / font_size
     - __sound__
       - music
       - No music
     - __font_size__
       - small
       - medium
       - large
 - This 2 by 3 design would result in a total of 6 treatment combinations.        










































































