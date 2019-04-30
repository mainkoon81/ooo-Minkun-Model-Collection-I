# Basics
• **Regression:** the target feature takes continuous values (real numbers, integers) 

• **Classification:** the target feature takes discrete values (labels, levels)

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
 - 1> Random Component(**distribution of the Response**): `exp()` 
   - The random component copes with the errors.
   - For the response variable, for example, the distribution that describes the **randomness** has to come from a particular family of distributions called an exponential family. Normal is a member of the exponential family of course.
 - 2> systematic component(**linear predictor**)
   - This is what we want to model. The systematic component is the Linear Component with the coefficients and predictors.    
 - 3> **link function** 
   <img src="https://user-images.githubusercontent.com/31917400/54211242-e3facd00-44d8-11e9-8975-0654c30fbc9a.jpg" />

   - This connects the `E[Y]`(from the **exponential family distribution**) to the **linear predictor** component.
     - It allows us to analyze the linear relationship b/w predictors and the **mean** of the response variable(because CLT says any mean follows Gaussian?) 

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
<img src="https://user-images.githubusercontent.com/31917400/53844949-4286ef00-3f9f-11e9-9ddc-ea0dfcb819c5.jpg" /> So here, we're transforming the `mean(or probability) value of the distribution`. We're not transforming the Response variables themselves.
 - For categoric data
 - For **binary** classification
 - The dependent variable must be **categorical**, and the **explanatory variables can take any form**.
 - But before fitting on the data, we need to convert the categorical predictors into the numeric(using `pd.get_dummies(series)[which level is '1'?]`). This is called:
   - `One-Hot Encoding` when there is no ordinal levels (red, blue,..=> check(1), uncheck(0),..)
   - `Integer Encoding` when there is ordinal levels (first, second,..=> 1,2..) 
   - http://pbpython.com/categorical-encoding.html

-----------------------------------------------------------------------------------------------
## [Background Study]
> Do you know **likelihood**? **log odd-ratio** ?
 - Prabability: To find probabilities, Fit data to the certain **distribution** we know: `P(θ|x,x,x,x)`
 - Likelihood: To find popultion parmeters, Fit distribution to the certain **data** we know: `P(x,x,x,x|θ)`
<img src="https://user-images.githubusercontent.com/31917400/53687390-0566f680-3d2c-11e9-81c2-a76eb822c462.jpg" />

> What can we do with `log(odd_ratio)` ?
<img src="https://user-images.githubusercontent.com/31917400/53687104-3cd3a400-3d28-11e9-95ae-a59e075e5b31.jpg" />

## 1. `log(odd_ratio)/SD` tells us if there is a strong/weak **relationship** between **two binary variables**.  
 - There are 3 ways to determine if the `log(odd_ratio)` is statistically significant..i.e determine the `p-value` for the significance of that relationship.    
   - Fisher's Exact Test
   - Chi-Sqr Test(to find P-value) 
   - Wald Test(to find P-value and Confidence Interval in Normal)
 - **Wald test:** 
   - It takes advantage of the fact that `log(odd_ratio)`, just like `log(odd)`, follows **`Normal`**.   
   <img src="https://user-images.githubusercontent.com/31917400/53688530-5206fd80-3d3d-11e9-8ce0-336fca1e8746.jpg" />

## 2. Maximum Likelihood Estimation is...
<img src="https://user-images.githubusercontent.com/31917400/54281684-29c79c00-4592-11e9-9c90-b4293a278078.jpg" />

To find the “most likely” parameter value that corresponds to the desired probability distribution of the sample data.
 - Properties(BLUEness)
   - sufficiency(complete information about the parameterof interest contained in its MLE estimator); 
   - consistency(true parametervalue that generatedthe data recovered asymptotically, i.e.for data of sufficientlylarge samples);
   - efficiency(lowest-possible variance of parameterestimatesachieved asymptotically); 
   - parameterization invariance(same MLE solutionobtainedindependent of the parametrization used).  

> ex_1. Normal
<img src="https://user-images.githubusercontent.com/31917400/54283654-61d0de00-4596-11e9-9e98-ac38777b5e66.jpg" />

> ex_2. OLS Regression
 - Let's say we apply the maximum likelihood principle to regression to find model coefficients.  
 <img src="https://user-images.githubusercontent.com/31917400/54300607-1e399c80-45b5-11e9-8346-566848cb8fd6.jpg" />

 - For this regression model, the maximum likelihood estimators are identical to the least squares estimators. The estimator of **σ** will, however, be slightly different.

> [Nagative Truth]
 - MLE does not eliminate sampling noise, or give us the truth. It’s just a decent estimator .

-------------------------------------------------------------------------------------------------------------------------------------

## Back to the main topic,
### 0. Classification and LogisticRegression
To attempt classification, one method is to use **linear regression** by mapping all **Y** greater than 0.5 as `1` and all less than 0.5 as `0`. However, this method doesn't work well because **classification is not actually a linear function**. It's a squiggle line. 

> PREDICTION: based on the line best cut the data, we can guess 'pass/fail' of new student.
<img src="https://user-images.githubusercontent.com/31917400/39021406-93efa878-4428-11e8-8bac-04d841fbbf16.jpg" />

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
 <img src="https://user-images.githubusercontent.com/31917400/54194105-62dd0f00-44b3-11e9-8fd3-b47f62067121.jpg" />

### III. Cost Function and Gradient Descent for LogisticRegression (Parameter Estimation_3)
To fit the model(to find the parameter), 
 - We use Maximum likelihood Estimation?
 - We know that in order to fit the model, we can Minimize the Cost Function, using Gradient Descent. 
   - the cost function can be derived from the principle of maximum likelihood estimation as well. 
 - There are many other complicate **Optimization algorithms** available to fit the model other than MaximumLikelihood, Gradient Descent...
   - Conjugate	gradient
   - BFGS, L-BFGS
<img src="https://user-images.githubusercontent.com/31917400/53877718-d8ebfc80-4001-11e9-8204-fc861dba512b.jpg" />

### IV. Model Evaluation for LogisticRegression
#### Q. How do we know the **fitted line of the highest log-likelihood value we discovered** is truley the best?  

 - In OLS regression, R_Squared & P_value are calculated using the **residuals**. 
   <img src="https://user-images.githubusercontent.com/31917400/53695305-a946b580-3db1-11e9-9c7d-4a73832a32e6.jpg" />

 - In Logistic regression, R_Squared & P_value are calculated using **log(likelihood)** 
   <img src="https://user-images.githubusercontent.com/31917400/54246226-00742500-452c-11e9-81f3-f8542094de9a.jpg" />

   - For independent observations, the likelihood is the product of the probability distribution functions of the observations. `-2` is a convention. 
   - McFadden's psuedo R-Squared & P_value
   # R^2
     - For R-Squared, instead of using `SSR/SST`, we use `[LL(Null) - LL(fit)] / [LL(Null) - LL(Saturated)]`. **This is not Chi-Sqr**. 
       - **LL(fit)** referring the **fitted line's highest log-likelihood value**. 
       - **LL(Saturated)** is not useful here since there is no data reduction (NO.of parameters = NO.of observations). It's considered as Maximum achievable log(likelihood). But it's a baseline for comparison to other model fits. 
   # P-value    
     - For P-value, `-2*[LL(Null) - LL(fit)]` = **Chi-Sqr value** with `df = the difference in the NO.of parameters in the two models`.
       - In the worst case, `LL(fit) = LL(Null)`, then Chi-Sqr value is 0, so P-value = 1 (area under the curve). 
       - In other cases, `LL(fit) > LL(Null)`, then Chi-Sqr is greater than 0, so P-value becomes smaller. 
       <img src="https://user-images.githubusercontent.com/31917400/53702648-1d5c7a00-3e01-11e9-9596-ea6d96d2e05d.jpg" />

#### Q. What about each Coefficient ?
When "likelihood" is mentioned..it's about overall model?..
 - Let's say `log(likelihood(odd))` is used to calculate **overall model-fit** significance (R^2) 
 - Let's say `log(odd_ratio)/SD` is a **Wald statistics** to tell if there is a strong/weak **relationship** between **two binary variables**.
 - Let's say `-2*log(likelihood_Ratio)` is a **Deviance statistics** that used for **Model Comparison**. 
   - Ho: Nested_Model = Full_Model
     - ..see if the full model is a rubbish..and not significant.  
   - Ho: Null_Model = Fitted_Model
     - ..see if the fitted model is a rubbish..and not significant. 
 - Interestingly, `log(odd_Ratio)` is useful to understand our **fitted_model-coefficients** !!!
   - Let's say we have a single fitted_model.
   - The **odd_ratio** measures the odds of an event compared to the odds of another event. 
     - Odd_Ratio ranges from 0 to ∞
     - `0 < Odd_Ratio < 1` indicate the **odds of Event 2** are greater.
     - `1 < Odd_Ratio < ∞` indicate the **odds of Event 1** are greater. 
   - w/o using likelihood, `log(odd_ratio)` means literally the log of difference b/w two odds produced by the same fitted_logit model equation.
     <img src="https://user-images.githubusercontent.com/31917400/54269194-d09d3f80-4574-11e9-8137-18cb3efbea30.jpg" />
   
   - This directly helps us understand the model coefficients. 
     <img src="https://user-images.githubusercontent.com/31917400/54277392-a6a14880-4587-11e9-9bf1-c93a4a43422f.jpg" />

### V. Goodness of Fit test with Deviance Statistics
 - Pearson Test
 - **Deviance Test**
 - HosmerLemeshow-C / H

What's Saturated Model used for? 
 - For Deviance statistics!
<img src="https://user-images.githubusercontent.com/31917400/53702415-90b0bc80-3dfe-11e9-8f75-908c7e6bb11b.jpg" />

## Deviance statistics:`Null Deviance - Residual Deviance` 
 - Comparing to **Chi-Sqr** (df = No.of parameters in `Proposed model` - No.of parameters in `Null model`) 
   - `-2*[LL(Null) - LL(Saturated)]` - `-2*[LL(fit) - LL(Saturated)]` = `-2*log(likelihood_Ratio)`
 - **Null Deviance**  
   - **Is `Null model` is significantly different from Saturated model?** 
   - Statistics:`-2*[LL(Null) - LL(Saturated)]` and this gives us **P-value**.
   - by comparing to **Chi-Sqr** (df = No.of parameters in Saturated model - No.of parameter in Null model) 
 - **Residual Deviance**  
   - **Is our `Proposed model` is significantly different from Saturated model?** 
   - Statistics:`-2*[LL(fit) - LL(Saturated)]` and this gives us **P-value**.
   - by comparing to **Chi-Sqr** (df = No.of parameters in Saturated model - No.of parameter in Proposed model)

What's Deviance Statistics used for?
 - To select model and coefficients
 - To study the effect of each predictor on the response variable. 
 - To distinguish Unbalance Data (Few responses of one type) ?

What's Deviance **Residual** in logistic regression used for?
 - To evaluate model fitting
 - Try to find large residual !! 
   - to detect outliers!! obv as pain in the ass.(not modeled well)
 
What else?
 - Akaike Information Criterion (**AIC**) 
   - Penalizes model for having many parameters 
   - AIC = `Deviance + 2*p` (where p = number of parameters in model)
 - Bayesian Information Criterion (**BIC**) 
   - BIC = `-2*LL(Null/fit) + ln(n)*p` (where p = number of parameters in model, and n = number of observations 
   - AKA the Schwartz Information Criterion (SIC)
 - Find the best subset (Tests all combinations of predictors)
 - Algorithmic Method
   - Stepwise Selection (Combination of forward and backward selection) 
     - 1. Fit each predictor variable as a single predictor variable and determine fit
     - 2. Select variable that produces best fit and add to model
     - 3. Add each predictor variable one at a time to the model and determine fit
     - 4. Select variable that produces best fit and add to the model
     - 5. Delete each variable in the model one at a time and determine fit
     - 6. Remove variable that produces best fit when deleted
     - 7. Return to Step Two
     - Loop until no variables added or deleted improve the fit
 - Outlier detection
   - Studentized Residual Plot (against predicted values)
   - Deviance Residual Plot (against predicted values)
     - Looking for “sore thumbs”, values much larger than those for other observations

### VI. Logistic Regression Assumption
> Logistic regression does not make many of the key assumptions of general linear models that are based on OLS algorithms: (1) logistic regression does not require a linear relationship between the dependent and independent variables, (2) the error terms (residuals) do not need to be normally distributed, (3) homoscedasticity is not required, and (4) the dependent variable in logistic regression is not measured on an interval or ratio scale. However, logistic regression still shares some assumptions with linear regression, with some additions of its own. 
 - __ASSUMPTION OF APPROPRIATE OUTCOME STRUCTURE:__ To begin, one of the main assumptions of logistic regression is the appropriate structure of the outcome variable. 
   - Binary logistic regression requires the dependent variable to be binary. 
   - Ordinal logistic regression requires the dependent variable to be ordinal. 
 - __ASSUMPTION OF OBSERVATION INDEPENDENCE:__ Logistic regression requires the observations to be independent of each other.  In other words, the observations should not come from repeated measurements or matched data. 
 - __ASSUMPTION OF THE ABSENCE OF MULTICOLLINEARITY:__ Logistic regression requires there to be little or no multicollinearity among the independent variables.  This means that the independent variables should not be too highly correlated with each other. 
 - __ASSUMPTION OF LINEARITY OF INDEPENDENT VARIABLES AND LOG ODDS:__ Logistic regression assumes linearity of independent variables and log odds. Although this analysis does not require the dependent and independent variables to be related linearly, it requires that the independent variables are linearly related to the log odds. 
 - __ASSUMPTION OF A LARGE SAMPLE SIZE:__ Finally, logistic regression typically requires a large sample size.  A general guideline is that you need at minimum of 10 cases with the least frequent outcome for each independent variable in your model. 
   - For example, if you have 5 independent variables and the expected probability of your least frequent outcome is 0.1, then you would need a minimum sample size of 500 (10*5 / 0.1). 


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
 - bias VS variance
 - Overfitting..high variance..How to solve?
   - **Regularization** (when we have to keep all features)
     - **L2:** Keep all features, but reduce magnitude/values of parameters.    
     - **L1:** Feature Selection
   - **Boosting & Bagging**
 - Validation and Assessing model performance
 
> Bias & Variance in ML
 - Bias: the inability for a model to capture the true relationship in the dataset
 - Variance: the difference in model fits between dataset(training vs test) 
 - The ideal: **low bias** & **`low variance`**
> Straight line and Squiggly line:
 - Straight line has:
   - medium(relatively high) **bias** (in training)
   - **`low variance`** (b/w training and testing) means...**consistency** in model performance! 
 - Squiggly line has:
   - low bias (in training)
   - high variance (b/w training and testing) means...**inconsistency** in model performance!..so we can say "Overfitting!".
<img src="https://user-images.githubusercontent.com/31917400/54045989-f6b88d80-41ca-11e9-9854-9b75a4b198cb.jpg" />

## Overfitting 
<img src="https://user-images.githubusercontent.com/31917400/53956695-36e91480-40d4-11e9-9183-a73e4c78e9e5.jpg" />

### `Regularization` in Cost Function
<img src="https://user-images.githubusercontent.com/31917400/53965694-969dea80-40e9-11e9-908a-44afe2a488e1.jpg" />

 - __Idea:__ Take the complexity of the model into account when we calculate the error. If we find a way to increment the error by some function of the `coefficients`, it would be great because in some way the complexity of our model will be added into the error so a complex model will have a larger error than a simple model. We respect that the simpler model have a tendency to generalize better.
 - **L1 regularization** is useful for `feature selection`, as it tends to turn the less relevant weights into zero.
 
## Q. so why do we punish our coefficient(slope)?
 - When our model looks so complex, the model sometimes suffer from **high variance**, so...we punish it by increasing bias(diminishing slopes). 
 - In the presence of multicollinearity b/w our coefficients, the model will suffer from **high variance**, so...we punish it by increasing bias(diminishing slopes).   
 - For every case, we have to tune how much we want to punish complexity in each model. This is fixed with a parameter `λ`. 
   - If having a small `λ`: multiply the complexity error by a small `λ` (won't swing the balance - "complex model is fine".)
     - A model to send the rocket to the moon or a medical model have very **little room for error** so we're ok with some complexity.
   - If having a large `λ`: multiply the complexity error by a large `λ` (punishes the complex model more - "simple model".)
     - A model to recommend potential friends have **more room for experimenting** and need to be simpler and faster to run on big data.
<img src="https://user-images.githubusercontent.com/31917400/39946131-e7c5a1da-5564-11e8-83f5-3f2e8e7c021d.jpg" />

**`λ`**, a regularization parameter controls a **trade off** between two different goals
 - __Goal 01:__ Fit the training set well to avoid **underfitting**
   - But if λ is too `small`, then Model coefficients are still huge, and results in overfitting a.k.a **complex model** (fails to fit even the training set).
 - __Goal 02:__ Keep these parameters small to avoid **overfitting** (keeping the hypothesis model relatively simple)
   - But if λ is too `large`, then diminishes and makes these parameters `close to zero`, and results in underfitting a.k.a **simple model** (fails to fit even the training set).

## Regularizing model's Coefficients can balance bias and variance.
So eventually we will find the model's coefficient from minimizing the cost function with the penalty term. 
 - Ridge (in case that all coefficients need to survive)
 - Lasso (in case that you can kill some coefficients)
 - ElasticNet
## so..Do you want to `increase bias` to reduce variance ? 
### 1. RidgeRegression(L2 Regularization:`SUM(β^2)`)
Find a new model that doesn't fit the **training data** that much by introducing a **small `bias`** into the line-fitting, but in return, we get a significant drop in variance. By starting with slightly worse fit, the model can offer better long term prediction! 
 - OLS regression's cost function is about minimizing the **SSE**
 - Ridge regression's cost function is about minimizing the **SSE** + `λ*slope^2`.
   - `slope^2` in the cost function means **penalty**
   - `λ` means **severity** of the penalty...`λ` is always equal to or greater than 0 

 - 1>Fitting and penalty(for numeric Response, numeric Predictors)
   - The larger `λ` get, ..slope get smaller, thus..the response becomes less, less sensitive to X-axis: **predictors**. 
 <img src="https://user-images.githubusercontent.com/31917400/54071365-90cc1480-4263-11e9-975e-9d4337c02df1.jpg" />

 - 2>Fitting and penalty(for numeric Response, categorical Predictors)
   - The larger `λ` get,..shrink slope(β1) down, thus..the response becomes less, less sensitive to X-axis: **disparity b/w classes**.   
 <img src="https://user-images.githubusercontent.com/31917400/54073959-e95fd980-4284-11e9-89ca-280b26c22dec.jpg" />

 - 3>Discussion on **Ridge penalty**
   - Ridge can also be applied to **Logistic regression** of course..but its cost function is like: `SUM(likelihood) + λ*slope^2`
   - Ridge can also be applied to **Multiple regression** of course..but its cost function is like: `SSE + λ*Σ(slope^2)`
   - In penalty term, we **ignore the intercept coefficient** by convention because every coefficient except the intercept is supposed to be scaled by measurements.
   - Then How to find `λ`? Try a bunch of `λ` and use **Cross Validation** to determine which one (b/w many models produced from: "cost function on training set--> coeff-estimation--> fit on testing set") results in the **lowest variance**.
   - **#** Let's say we have a **high dimensional** dataset(such as 1000 features) but **don't have enough samples**(such as 100 records). Our model have 1000 predictors, so we need more than 1000 datapoints to find the model_coefficients(each point per each dimension). so what should we do? **Do Ridge**!!!
     - It turns out that by adding the **penalty**, we can solve for all parameters with fewer samples. but how? 

### 2. LassoRegression(L1 regularization:`SUM(|β|)`)
Ridge cannot set coefficients to '0' while Lasso can shrink coefficients to '0', thus can be useful for **feature selection**. 
 - If `λ`= ∞, feature's slope = 0
 <img src="https://user-images.githubusercontent.com/31917400/54079029-d9b7b380-42cb-11e9-8b3d-cbe16ee6262d.jpg" />












































































