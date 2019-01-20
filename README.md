# Study-V004-Statistical-Model-Collection


### 1> Basic Regression
__A. LinearRegression__ is perhaps the simplest way to relate a continuous response variable to multiple explanatory variables.  Regression is sometimes called a many-sample technique. This may arise from observing several variables together and investigating which variables correlate with the response variable. Or it could arise from conducting an experiment, where we carefully assign values of explanatory variables to randomly selected subjects and try to establish a cause and effect relationship. 
<img src="https://user-images.githubusercontent.com/31917400/48806376-042c3380-ed12-11e8-8f37-67ef2e4e9ce7.jpg" />

 - The common choice for prior on the σ2 would be an `InverseGamma`
 - The common choice for prior on the β would be a `Normal`. Or we can do a multivariate normal for all of the β at once. This is conditionally conjugate and allows us to do **Gibbs-Sampling**.
 - Another common choice for prior on the β would be a `Double Exponential`(Laplace Distribution):`P(β) = 1/2*exp(-|β|)`. It has a sharp peak at `β=0`, which can be useful for **variable selection** among our X's because it'll favor values in your 0 for these βssss. This is related to the popular regression technique known as the `LASSO` ShrinkRegression(for more interpretable model with less variables). 
 - > **Interpretation** of the β coefficients: 
   - While holding all other X variables constant, if `x1` increases by one, then the `mean of y` is expected to increase by **β1**. That is, **β1** describes how the `mean of y` changes with changes in `x1`, while **accounting for all the other X variables**. 
 - > slope and intercept  
   - slope: `Sxy/Sxx` where `Sxx:(n-1)var(x)` and `Sxy:(n-1)cov(x,y)`
   - intercept: `E[y] - slope*E[x]`
   - SE(slope) = `sqrt((1/Sxx)*MSE)`
   - SE(intercept) = `sqrt((1/n + E[x]^2/Sxx)*MSE)`
 - > **Assumptions:** 
   - Given each X:`x1, x2,..` and independent Y:`y1, y2,..`: Multiple LM(many samples or columns)
     - `y1, y2,..` share the same variance `σ2`...and... `var(Y)`=`var(Xβ)+var(ϵ)`=`var(ϵ)`=`σ2` because fitted value..`var(Xβ)=0` ?
       - We know `var(Y)`is`MST`, `var(Xβ)`is`MSR`, and `var(ϵ)`is`MSE`.
         - `MST` is about `Y vs E[Y]`
         - `MSR` is about `fitted value vs E[Y]`
         - `MSE` is about `Y vs fitted value`..so Residuals.
         - `df for var(ϵ)`= n-(k+1), `df for var(Y)`= n-1, `df for var(Xβ)`= k
           - **k** is No.of predictors(`X`). `MSR` is interested in this. 
           - **k+1** is No.of coefficients(`β`). `MSE` is interested in this.        
       - We know `R^2` = `SSR/SST` = `1 - SSE/SST` : the model explains the ? % of variance in observations.
         - The `adjusted R^2` is `1 - var(ϵ)/var(Y)` which is the penalized by dividing them by `df`.
       - `R^2` cannot determine whether the coefficient estimates and predictions are biased, which is why we must assess the residual plots.          
         - In the plot of residual, `var(ϵ)` should be a constant: homoscedasticity..otherwise, you might want a weighted_LS solution...
         - The chart of `fitted value VS residual` should shows a flat line...
         - Our residuals should be Normally distributed..
     - E[**ε**] = 0
     - var(**ε**) = `σ2`
       - Are you sure? MSE = MST = `σ2`?? Yes, we hope `MSR = 0`. We wish our model(fitted_value) is the perfect E[Y]. 
     - cov(**ε**, fitted_value) = 0
     - In summary... `Y=Xβ+ϵ`; where `Y∼N(Xβ, σ2)` and `ϵ∼N(0, σ2)`.. E[Y] wish to be the model(wish MSE=MST), and E[error] wish to be zero.. 
   - If they are not met, **Hierachical model** can address it.    


__B. LogisticRegression__ is 





__C. PoissonRegression__ is 




__D. ANOVA with LinearRegression__ is used when we have **categorical explanatory variables** so that the observations`Y` belong to groups. In ANOVA, we compare the variability of responses(Y) `within groups` to the variability of responses(Y) `between groups`. If the variability **between groups** is large, relative to the variability within the groups, we conclude that there is a `grouping effect`. One Factor can have `2 levels` and the other can have `many levels`. For example, low and high or true and false. Or they can have many levels. 
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
We have assumed that all the observations were independent so far, but there is often a natural grouping to our data points which leads us to believe that some observation pairs should be more similar to each other than to others. For example, let's say a company plan to sample 150 test products for quality check, but they do 30 from your location and then 30 from 4 other factory locations(30 from each of 5 locations). We might expect a product from your location to be more similar to another product from your batch than to a product from another locations' batch. And we might be able to account for the likely differences between products in our Poisson model by making it a hierarchical model. That is, your lambda is not only estimated directly from your 30 products, but also indirectly from the other 120 products leveraging this hierarchical structure. Being able to account for relationships in the data while estimating everything with the single model is a primary advantage of using hierarchical models. And most Bayesian Models are hierarchical. 
<img src="https://user-images.githubusercontent.com/31917400/48874302-7ff9af00-edea-11e8-835e-ff0b7ff2f098.jpg" />

How we might use hierarchical modeling to extend a linear model? 
<img src="https://user-images.githubusercontent.com/31917400/48876484-3911b680-edf6-11e8-892b-ec6e8ed8b284.jpg" />

> Application
 - **Mixture models** provide a nice way to build nonstandard probability distributions from simper distributions, as well as to identify unlabeled clusters/populations in the data. Mixture models can be formulated **hierarchically**, allowing us to estimate unobserved(latent) variables in a technique called `data augmentation`.
 - **GLM** generalize normal linear regression models in the sense that the likelihood belongs to a more general class of distributions. `Data augmentation` techniques similar to those used for mixture models make GLMs amenable to **hierarchical modeling**. 
 - Observations from distinct **spatial locations** can exhibit dependence(just as observations collected across time are often correlated) For example, we might expect a measurement at location x to be more similar to `measurement y` 5 meters away than to `measurement z` 100 meters away. `State-space models` and `Non-parametric models` for response surfaces are common for spatial data. 
 - **DeepLearning models** involve layers of “neurons” that separate inputs from outputs, allowing nonlinear relationships. These **intermediate nodes** can be thought of as `latent variables` in a **hierarchical probabilistic model**, although Bayesian inference of neural networks is uncommon. 
 - **Bayesian Non-parametric models** move beyond inference for `parameters` to inference for `functions and distributions`. Finite-dimensional representations of the necessary priors often appear as **hierarchical models**. 2 of the most popular non-parametric priors are the `Gaussian process prior`(typically used as a prior on continuous functions), and `Dirichlet process prior`(as a prior on probability distributions). 









### 3> Latent Variable Models
Latent variable is just a random variable which is unobservable to you nor in training nor in test phase. This is the variable you can't just measure with some quantitative scale. 
<img src="https://user-images.githubusercontent.com/31917400/48974117-ebd85380-f046-11e8-913b-f788ec6bf63f.jpg" />

> __[Note]:__  
 - Latent variable can make things **easily interpretable** as we, for example, can estimate his intelligence on the scale from one to 100 although you don't know what the "scale" means. However, you can compare each data point according to this scale. Training latent variable model relies on a lot math.
 - Sometimes adding latent variables **restrict(overly simplify) your model** too much. If, for example, a student is doing 2 tests in the same day, it doesn’t make sense to assume that these two grades are caused only by his intelligence and doesn’t influence each other directly. Even if we know that he is very smart, if he failed the first test he is more likely to fail the second one because he may have a headache or maybe he didn’t have time to prepare the day before.

### Probabilistic Clustering with latent variable
__Hard/Soft Clustering:__ Usually clustering is done in a hard way, so for each data point we assign its **membership**. Sometimes, people do soft clustering. So instead of assigning each data point a particular membership, we will assign each data point from **probability** distributions over clusters(for example, 40% probability to belong to the A_cluster, and 60% probability to belong to the B_cluster, and 0% to the C_cluster). That is, instead of just assigning each data point a particular cluster, we assume that each data point belongs to every cluster, but with some **different probabilities**: `P(cluster_idx|x)` instead of `cluster_idx = f(x)`. Whyyyyy?  
 - 1. For handling **missing data**
 - 2. For tunning **hyperparameters**   




















































