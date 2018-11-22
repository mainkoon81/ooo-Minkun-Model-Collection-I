# Study-V004-Statistical-Model-Collection


### 1> Basic Regression
__A. LinearRegression__ is perhaps the simplest way to relate a continuous response variable to multiple explanatory variables. This may arise from observing several variables together and investigating which variables correlate with the response variable. Or it could arise from conducting an experiment, where we carefully assign values of explanatory variables to randomly selected subjects and try to establish a cause and effect relationship. 
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


__B. LogisticRegression__ is 





__C. PoissonRegression__ is 




__D. ANOVA with LinearRegression__ is used when we have **categorical explanatory variables** so that the observations`Y` belong to groups. In ANOVA, we compare the variability of responses(Y) `within groups` to the variability of responses(Y) `between groups`. If the variability **between groups** is large, relative to the variability within the groups, we conclude that there is a `grouping effect`. Factors can have `2 categories` or `many levels`. For example, low and high or true and false. Or they can have many levels. 
 - Let's say we are going to conduct an online marketing experiment.
   - we might experiment with two factors of website design - sound / font_size
     - factor01:__sound__
       - music
       - No music
     - factor02:__font_size__
       - small
       - medium
       - large
   - This 2 by 3 design would result in a total of 6 treatment combinations.        
 - > One factor Model: How ANOVA is related to LinearRegression ? 
 <img src="https://user-images.githubusercontent.com/31917400/48838897-a7686180-ed81-11e8-8927-df3f0f0ce8d4.jpg" />
 
 - > Two factor Model: cell_means_model where we have different mean for each stream and combination ?  
 <img src="https://user-images.githubusercontent.com/31917400/48842101-46458b80-ed8b-11e8-933f-ac288ebb2aee.jpg" />

   - If the effect of factor A on the response changes between levels of factor B, then we would need more parameters to describe how that mean changes. This phenomenon is called **interaction** between the factors.


### 2> Hierarchical Model
We have assumed that all the observations were independent so far, but there is often a natural grouping to our data points which leads us to believe that some observation pairs should be more similar to each other than to others. For example, let's say a company plan to sample 150 test products for quality check, but they do 30 from your location and then 30 from 4 other factory locations(30 from each of 5 locations). We might expect a product from your location to be more similar to another product from your batch than to a product from another locations' batch. And we might be able to account for the likely differences between products in our Poisson model by making it a hierarchical model. That is, your lambda is not only estimated directly from your 30 products, but also indirectly from the other 120 products leveraging this hierarchical structure. Being able to account for relationships in the data while estimating everything with the single model is a primary advantage of using hierarchical models.
<img src="https://user-images.githubusercontent.com/31917400/48874302-7ff9af00-edea-11e8-835e-ff0b7ff2f098.jpg" />

How we might use hierarchical modeling to extend a linear model? 
<img src="https://user-images.githubusercontent.com/31917400/48876484-3911b680-edf6-11e8-892b-ec6e8ed8b284.jpg" />
































































