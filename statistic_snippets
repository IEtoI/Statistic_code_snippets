#collection of statistics in python
#INTERESTING PACKAGE TO CHECK OUT: SPM1D
#->http://www.spm1d.org
#ANCOVA - Analysis of covariance 
#This can be done to compare two regression lines with each other 
#Example 
"""
DC  EtOH  BEC   Condition  
-------------------------
BD  2.60  59.0  True   
HD  2.20  77.0  True   
HD  2.22  60.0  False 
... ...   ...   ...
"""
from statsmodels.formula.api import ols
# formula = 'BEC ~ EtOH' #simple regression
formula = 'BEC ~ EtOH * C(Condition)' #ANCOVA formula
lm = ols(formula, dataframe[dataframe.DC == 'HD'])
fit = lm.fit()
print(fit().summary())
#-------------------------------------------------------------------------------------------------
"""
TEST FOR NOMINAL DATA 
SUMMARIZED AS PERCENTAGE OR PROPORTION
Goodness-of-fit tests
Those tests are just when comparing frequencies of one nominal variable to theoretical expectations
E.g. offspring expected 3:1 ratio -> compare to observed offspring
"""
#Binomial test
#Use it when only 1 nominal variable with only 2 values
#example cat bats ball with pawns 10 times: 2 times left 8 times right 
#Q: Is there a significant preference for the right pawn? 
def binomdist(n,p,succ,tails=2,exact=False):
    #NOT RECOMENDED IF NULL HYP IS OTHER THAN 1:1!!!!
    """Function which calculates exact binomimal test
    Input: n: number of total bets, p: prob of bet, succ: smaller number of both possibilitis
    tails: 1- or 2- tailed, exact probability: False for as large or larger True for as large
    Output: probability"""
    
    import scipy.stats as ss

    hh = ss.binom(n, p)
    #Calculate prob for getting deviation as large OR LARGER therefore for loop 
    total_p = 0
    for k in range(1, succ + 1):  # DO NOT FORGET THAT THE LAST INDEX IS NOT USED
        total_p += hh.pmf(k)
    return total_p*tails

binomdist(20, 0.5, 4, tails=2)
#------------------------------------------------------------------------------------------------
from scipy.stats import chisquare
#important that sample size is large enough >500 otherwise not accurate
#chi square of certain observations and the corresponding expectations 
print(chisquare(f_obs=[70,79,3,4], f_exp=[156*0.54,156*0.4,156*0.05,156*0.01]))
#------------------------------------------------------------------------------------------------
import numpy as np
from scipy.stats import, chisquare
from scipy.stats.distributions import chi2
#G-test is comparable to chisquare test but can be used for more elaborate
#statistical designes such as repeated G-tests of goodness-of-fit
def gtest(f_obs, f_exp=None, ddof=0):
    """
    http://en.wikipedia.org/wiki/G-test
    The G test can test for goodness of fit to a distribution
    Parameters
    ----------
    f_obs : array
        observed frequencies in each category
    f_exp : array, optional
        expected frequencies in each category.  By default the categories are
        assumed to be equally likely.
    ddof : int, optional
        adjustment to the degrees of freedom for the p-value
    Returns
    -------
    chisquare statistic : float
        The chisquare test statistic
    p : float
        The p-value of the test.
    Notes
    -----
    The p-value indicates the probability that the observed distribution is
    drawn from a distribution given frequencies in expected.
    So a low p-value inidcates the distributions are different.
    Examples
    --------
    >>> gtest([9.0, 8.1, 2, 1, 0.1, 20.0], [10, 5.01, 6, 4, 2, 1])
    (117.94955444335938, 8.5298516190930345e-24)
    >>> gtest([1.01, 1.01, 4.01], [1.00, 1.00, 4.00])
    (0.060224734246730804, 0.97033649350189344)
    >>> gtest([2, 1, 6], [4, 3, 2])
    (8.2135343551635742, 0.016460903780063787)
    References
    ----------
    http://en.wikipedia.org/wiki/G-test
    """
    f_obs = np.asarray(f_obs, 'f')
    k = f_obs.shape[0]
    f_exp = np.array([np.sum(f_obs, axis=0) / float(k)] * k, 'f') \
                if f_exp is None \
                else np.asarray(f_exp, 'f')
    g = 2 * np.add.reduce(f_obs * np.log(f_obs / f_exp))
    return g, chi2.sf(g, k - 1 - ddof)

print(gtest(f_obs=[423,133], f_exp=[556*0.75,556*0.25]))
#-------------------------------------------------------------------------------------------------------
from scipy.stats import chisquare
import numpy as np
def randomizationGOF(f_obs, f_exp, iterations=10000, ddof=0):
    """
    Use when sample size is too small for chi-square/G-test
    Example: red vs. pink vs. white flowers
    Expectation: 1:2:1 
    You get 8 offspring -> 2:4:2 is expected but u have 5:2:1
    First chi-square statistic is calculated afterwards randomly draw samples with replacement
    and calculate chi-square stat. If calculated chi-stat > expected chi-stat in more than 5% of the 
    cases you can reject the Null hypothesis
    
    Input:
        
        f_obs: observations made
        f_exp: observations expected
    
    Returns: 
    
        P: percentage how often randomization trials produce chi-square stat larger than expected
    """
    counter = 0
    #calculate expected chi-square stat
    exp = chisquare(f_obs, f_exp)[0]  
    #draw n times from the choices with replacement
    n = sum(f_obs)    
    for i in range(iterations):
        draws = list(np.random.choice(len(f_obs),size=n,replace=True, p=[0.54,0.4,0.05,0.01]))
        counts = []
        for j in range(len(f_obs)):
            counts.append(draws.count(j)) 
        #np.unique(draws, return_counts=True)
        #d = list(zip(unique, counts))
        new = chisquare(counts, f_obs)[0]
        if new > exp:
            counter += 1

    return (counter/iterations)/100
    
print(randomizationGOF([70,79,3,4], [156*0.54,156*0.4,156*0.05,156*0.01]))
#---------------------------------------------------------------------------------------------------------
"""
Test of independence
When comparing frequencies of one nominal variable for different values of a second nominal variable
"""
#---------------------------------------------------------------------------------------------------------
#Chi-square of independence
from scipy.stats import chi2_contingency
"""example 
    species1    species2
area1  127         116
area2  99          67
area3  264         161

H0: Specie distribution is same for the areas
"""
row1 = [127, 116]
row2 = [99, 67]
row3 = [264, 161]
data = [row1, row2, row3]
print(chi2_contingency(data))
#-----------------------------------------------------------------------------------------------------------
#Fisher's exact test
#more accurate than chi-square test of independence for small sample sizes 
from scipy.stats import fisher_exact
"""
example:
        species1     species2
area1      16          8
area2      3          18

H0: Specie distribution is same for areas
"""
row1 = [16, 8]
row2 = [3, 18]
data = [row1, row2]
print(fisher_exact(data))
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
"""
TESTS FOR MEASUREMENT VARIABLES

Prelude:

Statistics of central tendency:
arithmetic mean:
    Sum of obs / num of obs
-> good for values that fit normal distribution
-> sensitive for outlier
-> not good for skewed data

geometric mean: 
-> rather usefull for economics not so usefull for biology

harmonic mean: 
-> some uses in engineering but so much in biology

median:
When values are sorted then the median is the value in the middle
-> useful for highly skewed distributions
-> when u have a population but cannot measure all values 

mode:
Most common value in a data set
-> continous data has to be grouped 

Spread:
range: 
largestValue - smallesValue

sum of squares: 
for i in measurement:
    (obs_i - mean)**2
bases of variance and standard deviation

coefficient of variation:
STDV / mean
-> can be used to compare amount of variation between samples w/o same mean
e.g. variance of pinkie finger vs little toe length

ERROR BARS:
Important to diverentiate between +- STDV and +- confidence limits

"""
#CONFIDENCE INTERVAL
import numpy as np
import scipy as sp
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h
#----------------------------------------------------------------------------------------------------
#STUDENTS T-TEST
#Used to compare means of two samples
#H0 means are equal
#Assumptions: Normal distribution, equal variance
# -> Mann-Whitney U-test for very non-normal distribution ttest_ind(a,b, equal_var=False)
#-> Welch's t-test if variance unequal

from scipy.stats import ttest_ind

a = [1,4,5,6,2,3,2,5]
b = [7,8,9,4,2,7,8,2]
print(ttest_ind(a,b))
#------------------------------------------------------------------------------------------------------------
#ONE WAY ANOVA
#Compares means of groups of measurement data
#one measurement variable and one nominal variable 
#H0 equal mean for all categories
#Assumptions: Normality, homoscedasticity (same variance)
#If Assumptions are violated -> Kruskal-Wallis test 
# If there are two or more nominal variables -> two-way anova or nested anova

#1. Choose between Model I and Model II ANOVA

#MODEL I ANOVA
#If groups are important; e.g. comparison of different drugs

#MODEL II ANOVA
#If groups are arbitrary: e.g. Look at random cells on 5 plates and measure size of nucleus 

#2. Testing the homogeneity of means

from scipy.stats import f_oneway

a = [1,5,4,6,8,7,4,5,6]
b = [5,4,6,5,4,7,8,5,4]
c = [1,2,4,5,4,7,8,5,2]
print(f_oneway(a,b,c))

#3. Planned comparisons of means
#Advantage: if you determine in advance which means you want to compare 
#           you do not have to adjust the p value for the tests you could have but did not do

#No p-value adjustment needed for orthogonal tests
#P-valúe adjustments 
#1. Bonferroni alpha/k
#2. Dunn-Sidak method sequential <- ONLY FOR PLANNED COMPARISON

def DunnSidak(k, alpha=0.05):
    """
    Input: k=number of comparisons
    Output: adjusted alphas 
    """
    adj_p = []
    while k!=0:
        adj_p.append(1-(1-alpha)**(1/k))
        k-=1
    return adj_p
    
print(DunnSidak(4))

#Post hoc Analysis for unplanned comparison after ANOVA
#Tukey HSD
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(endog=voter_age,     # Data
                          groups=voter_race,   # Groups
                          alpha=0.05)          # Significance level

tukey.plot_simultaneous()    # Plot group confidence intervals
plt.vlines(x=49.57,ymin=-0.5,ymax=4.5, color="red")

tukey.summary()              # See test summary

#this module also provides a lot of functions for multiple tests and p value correction 
from statsmodels.stats import multitest

#------------------------------------------------------------------------------------------------------------------------------
#Tests for normality 
from scipy.stats import normaltest, shapiro, norm
x = norm.rvs(loc=5, scale=3, size=100)
print(normaltest(x))
print(shapiro(x))
#--------------------------------------------------------------------------------------------------------------------------------
#Tests for equal variance

from scipy.stats import bartlett
x = norm.rvs(loc=5, scale=3, size=100)
y = norm.rvs(loc=5, scale=3, size=100)
data = [x,y]
print(bartlett(*data))
#When data is not quite normal better levene's test
"""
Three variations of Levene’s test are possible. The possibilities and their recommended usages are:
        ‘median’ : Recommended for skewed (non-normal) distributions>
        ‘mean’ : Recommended for symmetric, moderate-tailed distributions.
        ‘trimmed’ : Recommended for heavy-tailed distributions.
"""
from scipy.stats import levene
x = norm.rvs(loc=5, scale=3, size=100)
y = norm.rvs(loc=5, scale=3, size=100)
data = [x,y]
print(levene(*data))
#------------------------------------------------------------------------------------------------------------------
#Data Transformation:
#Log transformation 
#Transforms log-normal into normal distributions 
#Square-root transformation
#commonly used when variable is a count of something e.g. bacteria on a plate
#-------------------------------------------------------------------------------------------------------------------
"""
Tests for not normal distributed data
"""
#Kruskal-Wallis test <-> non parametric anologue of a one-way ANOVA
from scipy.stats import kruskal
print(kruskal(*data))
#Mann-Whitney U-test <-> non parametric anologue of T-Test
from scipy.stats import mannwhitneyu
print(mannwhitneyu(x,y))

#-----------------------------------------------------------------------------------------------------------------------
#Two way anova 
#https://www.marsja.se/three-ways-to-carry-out-2-way-anova-with-python/
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy import stats

def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    return aov
 
def omega_squared(aov):
    mse = aov['sum_sq'][-1]/aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
    return aov
#exmaple of tooth growth with variables supp dose supp:dose as explenatory and len as target 
#1. formulation of the model
formula = 'len ~ C(supp) + C(dose) + C(supp):C(dose)'
model = ols(formula, data).fit()
aov_table = anova_lm(model, typ=2)

"""
Statsmodels does not calculate effect sizes for us. unctions above can, again, be used 
and will add omega and eta squared effect sizes to the ANOVA table.
"""
#2, Add eta and omega squared
eta_squared(aov_table)
omega_squared(aov_table)
print(aov_table)

#3. Plot QQplot
res = model.resid 
fig = sm.qqplot(res, line='s')
plt.show()

#---------------------------------------------------------------------------------------------------------------------

