# Study-V004-Statistical-Model-Collection

# 0> Gradient Descent Method & Cost function
> Whenever you train a model with your data, you are actually producing some new values (predicted) for a specific feature. However, that specific feature already has some values which are real values in the dataset. We know the closer the predicted values to their corresponding real values, the better the model. 
# ε ?
 - **`Cost function`** needed to be minimized.
   - To measure how close the **predicted values** are to their **corresponding real values**. We also should consider that the **weights of the trained model** are responsible for accurately predicting the new values. 
   - Imagine that our model is y = 0.9*X + 0.1, the predicted value is nothing but (0.9*X+0.1) for different Xs. So, by considering Y as real value corresponding to this x, the cost formula is coming to measure how close (0.9*X+0.1) is to Y. We are responsible for finding the better weight (0.9 and 0.1..so **PARAMETERS**) for our model to come up with a lowest cost (or closer predicted values to real ones). 
# θ ?   
 - **`Gradient descent` to update PARAMETERS**.
   - As an optimization algorithm, its responsibility is to find the minimum cost value in the process of trying the model with different weights or indeed, **updating the weights**. 
   - It is made out of **`derivitives of Cost function`**.  
   - We first run our model with some initial weights and gradient descent updates our weights and find the cost of our model with those weights in thousands of iterations to find the minimum cost. One point is that gradient descent is not minimizing the weights, it is just updating them.

> (`NO.of parameter = 1`)
<img src="https://user-images.githubusercontent.com/31917400/53805999-a54a9d00-3f43-11e9-88c9-ae300355ef8a.jpg" />

> (`NO.of parameter > 1`) with Multiple features 
<img src="https://user-images.githubusercontent.com/31917400/53817587-b56f7600-3f5d-11e9-9e04-e9b1b21bbfe9.jpg" />

--------------------------------------------------------------------------------------------
# 1> GLM
> Transformations in LM are often **hard to interpret**( our model codefficient). Although there are some interpretable transformations, natural logarithms in specific, but they aren't applicable for negative or zero values(`ln(x) is defined only for x > 0`). Plus what if we encounter some moment where we are required to respect the original data without transformation? 

> What if, in the LM, the normal distribution is putting a lot positive probability on **negative values** even though you know your response has to be **positive** ? 

GLM involves three components:
 - 1> Random Component(**distribution**): `exp()` 
   - The random component copes with the errors.
   - For the response variable, for example, the distribution that describes the **randomness** has to come from a particular family of distributions called an exponential family. Normal is a member of the exponential family of course.
 - 2> systematic component(**linear predictor**)
   - This is what we want to model. The systematic component is the Linear Component with the coefficients and predictors.    
 - 3> **link function** 
   - This connects the `meaning of the response`(from the **exponential family distribution**) to the **linear predictor**.

## Ok, start with basics. 
## A. Linear Regression (NORMAL response): 
In this setting, we have data(Response variable) that comes from **Gaussian** distribution. 
 - Regression is sometimes called a **many-sample technique**. This may arise from observing several variables together and investigating which variables correlate with the response variable. Or it could arise from conducting an experiment, where we carefully assign values of explanatory variables to randomly selected subjects and try to establish a cause and effect relationship. 
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
   - If they are not met, **`Hierachical model`** can address it.    

## [B]. Logistic Regression (Y/N response):
In this setting, we have data(Response variable) that are `0/1` so binary, so it comes from **Bernoulli** distribution.  
<img src="https://user-images.githubusercontent.com/31917400/53844949-4286ef00-3f9f-11e9-9ddc-ea0dfcb819c5.jpg" /> So here, we're transforming the `mean(or probability) value of the distribution`. We're not transforming the Response variables themselves. That's the neat part of generalization in our models.

__[Background]__
> Do you know **likelihood**? **log odd-ratio** ?
 - Prabability: Fit data to the certain **distribution** we know.
 - Likelihood: Fit distribution to the certain **data** we know.
<img src="https://user-images.githubusercontent.com/31917400/53687390-0566f680-3d2c-11e9-81c2-a76eb822c462.jpg" />

> What can we do with `log(odd_ratio)` ?
<img src="https://user-images.githubusercontent.com/31917400/53687104-3cd3a400-3d28-11e9-95ae-a59e075e5b31.jpg" />

`log(odd_ratio)` tells us if there is a strong/weak **relationship** between **two binary variables**.  
 - There are 3 ways to determine if the `log(odd_ratio)` is statistically significant..i.e determine the `p-value` for the significance of that relationship.    
   - Fisher's **Exact Test**
   - Chi-Sqr Test(to find P-value) 
   - Wald Test(to find P-value and Confidence Interval)
> Wald test: 
 - It takes advantage of the fact that `log(odd_ratio)`, just like `log(odd)`, follows Gaussian.   
<img src="https://user-images.githubusercontent.com/31917400/53688530-5206fd80-3d3d-11e9-8ce0-336fca1e8746.jpg" />

## Back to the main topic,
### 0. Classification and LogisticRegression
To attempt classification, one method is to use **linear regression** by mapping all **Y** greater than 0.5 as `1` and all less than 0.5 as `0`. However, this method doesn't work well because **classification is not actually a linear function**. It's a squiggle line. 
 - [Decision Surface]
   -  The decision surf is a **`property`** of the **hypothesis model** that is made out of parameters, but it is **NOT** a property of the dataset. Once we have particular values for the parameters, then that completely defines the decision surf! 
   - When we actually describe or plot the decision surf, we come down to the orginal features' dimension. (In logistic regression, it's the dimension of `X` and `Y = log(odd) = Xβ`).
   <img src="https://user-images.githubusercontent.com/31917400/53842158-9b9e5500-3f96-11e9-9abe-31030c46342e.jpg" />

 - [Multiclass Classification]
   - To make a prediction on a new x, pick the class ￼that maximizes the hypothesis model: hθ(x). 
   <img src="https://user-images.githubusercontent.com/31917400/53878654-5ca6e880-4004-11e9-8d29-3ff2c2b29823.jpg" />

### I. Model_Fitting for LogisticRegression (Parameter Estimation_1)
 - Maximum Likelihood
 - Cost Function & Gradient Descent
<img src="https://user-images.githubusercontent.com/31917400/53828506-c5934f80-3f75-11e9-80e1-20ade17da543.jpg" />

### II. Maximum Likelihood for LogisticRegression (Parameter Estimation_2)
 - In OLS regression, we use **Least squares Method** to fit the line(to find slope and intercept).  
 - In logistic regression, we use **Maximum Likelihood Method** to fit the line or to obtain the best **p = sigmoid function** = Logistic Function. We project all data pt onto the best fitting line, using `log(odd)`, and then translate these `log(odd)`s back to probabilities using `sigmoid`.  
 <img src="https://user-images.githubusercontent.com/31917400/53696183-82da4780-3dbc-11e9-9753-7d53bc134333.jpg" />

### III. Cost Function and Gradient Descent for LogisticRegression (Parameter Estimation_3)
To fit the model(to find the parameter), 
 - We use Maximum likelihood Estimation?
 - We know that in order to fit the model, we can Minimize the Cost Function, using Gradient Descent. 
   - the cost function can be derived from the principle of maximum likelihood estimation as well. 
 - There are many other complicate **Optimization algorithms** available to fit the model other than MaximumLikelihood, Gradient Descent...
   - Conjugate	gradient
   - BFGS, L-BFGS
<img src="https://user-images.githubusercontent.com/31917400/53877718-d8ebfc80-4001-11e9-8204-fc861dba512b.jpg" />

### IV. Goodness of Fit test for LogisticRegression
How do we know the **fitted line of the highest log-likelihood value we used** is truley the best?  
 - In OLS regression, R_Squared & P_value are calculated using the residuals. 
   <img src="https://user-images.githubusercontent.com/31917400/53695305-a946b580-3db1-11e9-9c7d-4a73832a32e6.jpg" />

 - In Logistic regression,
   - Pearson Test
   - Deviance Test
   - HosmerLemeshow-C / H
   - McFadden's psuedo R-Squared & P_value
     - For R-Squared, instead of using `SSR/SSE`, we use `[LL(Null) - LL(fit)] / LL(Null)`. **LL(fit)** refering the **fitted line's highest log-likelihood value**. 
     - For P-value, `2*[LL(Null) - LL(fit)]` = **Chi-Sqr value** with `df = the difference in the NO.of parameters in the two models`.
       - In the worst case, `LL(fit) = LL(Null)`, then Chi-Sqr value is 0, so P-value = 1 (area under the curve). 
       - In other cases, `LL(fit) > LL(Null)`, then Chi-Sqr is greater than 0, so P-value becomes smaller. 
       <img src="https://user-images.githubusercontent.com/31917400/53702648-1d5c7a00-3e01-11e9-9596-ea6d96d2e05d.jpg" />

### V. Goodness of Fit test with Deviance Statistics
What's Saturated Model used for? 
<img src="https://user-images.githubusercontent.com/31917400/53702415-90b0bc80-3dfe-11e9-8f75-908c7e6bb11b.jpg" />

What's Deviance Statistics used for?
 - we can get **P-value** and compute the **log-likelihood based R-Squares**. 
   - From the statistics:`Null Deviance - Residual Deviance` 
   - by comparing to **Chi-Sqr** (df = No.of parameters in Proposed model - No.of parameter in Null model) 
 - **Residual Deviance**  
   - **Is our `Proposed model` is significantly different from `Saturated model`?** 
   - Statistics:`2*[LL(Saturated model) - LL(Proposed model)]` and this gives us **P-value**.
   - by comparing to **Chi-Sqr** (df = No.of parameters in Saturated model - No.of parameter in Proposed model)
 - **Null Deviance**  
   - **Is `Null model` is significantly different from `Saturated model`?** 
   - Statistics:`2*[LL(Saturated model) - LL(Null model)]` and this gives us **P-value**.
   - by comparing to **Chi-Sqr** (df = No.of parameters in Saturated model - No.of parameter in Null model)

## [C]. Poisson Regression (COUNT response):
In this setting, we have data(Response variable) that are unbounded counts(web traffic hits) or rates(1-star, 2-star...), so they come from **Poisson** distribution. Of course we can approximate **Bin(p,n) with small p and large n** with this model or we can analyse a **contingency table data** as well. Poisson Regression is also known as **`Log-Linear Model`**. So here, we're transforming the `mean(or probability) value of the distribution`. Again, We're not transforming the Response variables themselves. That's the neat part of generalization in our models.
 - Linear regression methods (assume constant variance and normal errors) are not appropriate for count data.
   - Variance of response variable increases with the mean
   - Errors will not be normally distributed
   - Zeros are a headache in transformations
 <img src="https://user-images.githubusercontent.com/31917400/53890974-3b082a00-4021-11e9-956c-305c901fc010.jpg" />

 - 1.Model for **Count**: Response is a count `E[Y]`.  
 <img src="https://user-images.githubusercontent.com/31917400/53890962-33488580-4021-11e9-8314-de1f752e8566.jpg" />
 
 - 2.Model for **Rate**: Reponse is `E[Y/t]` where `t` is an **interval** representing time, space, etc.  
 <img src="https://user-images.githubusercontent.com/31917400/53891890-304e9480-4023-11e9-88cc-42a9328aa21d.jpg" />

### 0. Classification and PoissonRegression
### I. Model_Fitting for PoissonRegression
### II. Maximum Likelihood for PoissonRegression
### III. Goodness of Fit test for PoissonRegression
Confidence Intervals and Hypothesis tests for parameters: 
 - Wald statistics
 - Likelihood ratio tests
 - Score tests
### IV. Goodness of Fit test with Deviance Statistics
 - Pearson chi-square statistic
 - Deviance statistics
 - Likelihood ratio statistic
 - Residual analysis: Pearson, deviance, adjusted residuals
 - Overdispersion(the observed variance is larger than the assumed variance)

## [D]. ANOVA with LinearRegression:
It is used when we have **categorical explanatory variables** so that the observations`Y` belong to groups. In ANOVA, we compare the variability of responses(Y) `within groups` to the variability of responses(Y) `between groups`. If the variability **between groups** is large, relative to the variability within the groups, we conclude that there is a `grouping effect`. One Factor can have `2 levels` and the other can have `many levels`. For example, low and high or true and false. Or they can have many levels. 
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

## [E]. Other Issues
 - Overfitting..How to solve?
   - **Regularization** (when we have to keep all features)
     - Keep all features, but reduce magnitude/values of parameters.    
   - **Feature Selection**
 - bias VS variance
 - Diagnostics: Assessing model performance
 - Validation

## a) Overfitting 
<img src="https://user-images.githubusercontent.com/31917400/53956695-36e91480-40d4-11e9-9183-a73e4c78e9e5.jpg" />

### `Regularization` in Cost Function
<img src="https://user-images.githubusercontent.com/31917400/53965694-969dea80-40e9-11e9-908a-44afe2a488e1.jpg" />

**`λ`**, a regularization parameter controls a **trade off** between two different goals
 - __Goal 01:__ Fit the training set well to avoid **underfitting**
   - But if λ is too `small`...the Algorithm results in overfitting (fails to fit even the training set).
 - __Goal 02:__ Keep the parameters small to avoid **overfitting** (keeping the hypothesis model relatively simple)
   - But if λ is too `large`(penalizing the parameters too high), then we will end up with all of these parameters `close to zero`, so the algorithm results in underfitting (fails to fit even the training set).











# 2> Hierarchical Model
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








# 3> Latent Variable Models
Latent variable is just a random variable which is unobservable to you nor in training nor in test phase. This is the variable you can't just measure with some quantitative scale. 
<img src="https://user-images.githubusercontent.com/31917400/48974117-ebd85380-f046-11e8-913b-f788ec6bf63f.jpg" />

> __[Note]:__  
 - Latent variable can make things **easily interpretable** as we, for example, can estimate his intelligence on the scale from one to 100 although you don't know what the "scale" means. However, you can compare each data point according to this scale. Training latent variable model relies on a lot math.
 - Sometimes adding latent variables **restrict(overly simplify) your model** too much. If, for example, a student is doing 2 tests in the same day, it doesn’t make sense to assume that these two grades are caused only by his intelligence and doesn’t influence each other directly. Even if we know that he is very smart, if he failed the first test he is more likely to fail the second one because he may have a headache or maybe he didn’t have time to prepare the day before.

### Ex_01> Probabilistic Clustering with latent variable
__Hard/Soft Clustering:__ Usually clustering is done in a hard way, so for each data point we assign its **membership**. But sometimes, people do soft clustering. So instead of assigning each data point a particular membership, we will assign each data point from **probability** distributions over clusters(for example, 40% probability to belong to the A_cluster, and 60% probability to belong to the B_cluster, and 0% to the C_cluster). That is, instead of just assigning each data point a particular cluster, we assume that each data point belongs to every cluster, but with some **different probabilities**: `P(cluster_idx|x)` instead of `cluster_idx = f(x)`. Whyyyyy?  
 - 1. For handling **missing data**
 - 2. For tunning **hyperparameters**(meta-parameters...it can be the No.of clusters of course)
 <img src="https://user-images.githubusercontent.com/31917400/51439273-b69b5b00-1caf-11e9-99ee-a39f00c652bc.jpg" /> 
 
 - 3. For building a **generative model** of our data.. treating everything probabilistically, we may sample new data points from our model of the data.

### So let's do Soft Clustering!
> __GMM__ Gaussian Mixture Model: **How to fit it? How to find the model parameters?** 
The simplest way to fit a probability distribution is to use **maximum likelihood**. Find the parameters maximizing the likelihood(density)! 
<img src="https://user-images.githubusercontent.com/31917400/51492177-c3e84080-1da8-11e9-8386-e1ce3e4eb595.jpg" /> 

We introduce a latent variable.
<img src="https://user-images.githubusercontent.com/31917400/51533344-71576480-1e3a-11e9-8570-c0128a7cc197.jpg" /> 

[Note]: 
 - When choosing the best run (highest training log-likelihood) among several training attempts with different random initializations, we can suffer from **local maxima**. and its solution would be **NP-Hard**. 
 - We use EM-Algorithm to fit the distribution(such as Gaussian) to the **multi-dimensional** data(estimating the mean vector `μ_` and the covariance matrix `σ_`), not to the one-dimensional data. 
 - We can treat `missing values` as **latent variables** and still estimate the Gaussian parameters. But with one-dimensional data, if a data point has missing values, it means that we don’t know anything about it (its only dimension is missing), so the only thing that is left is to throw away points with missing data.
 - Note that we also don’t need EM to estimate the mean vector (e.i. we need it only for the covariance matrix) in the multi-dimensional case: since each coordinate of the mean vector can be treated independently, we can treat each coordinate as one-dimensional case and just throw away missing values. 

### General form of EM-Algorithm























































