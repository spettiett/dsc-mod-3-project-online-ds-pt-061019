#flatiron_stats
import numpy as np
import scipy.stats as stats
from scipy.stats import anderson
import matplotlib.pyplot as plt


def monte_carlo_test(var1, var2, popl, col):
    """Non-Normal Distribution - Non-Parametric Tests: Using Monte Carlo Test"""
    print(f"Non-Parametric Tests: Using Monte Carlo Test")
    print(f"_____")
    mean_diff = np.mean(var1) - np.mean(var2)

    sample_diffs = []
    counter = 0
    for i in range(1000):
        samp1 = popl.sample(replace=False, n=len(var1))
        samp2 = popl.drop(samp1.index,axis=0)
        sample_diff = samp1[col].mean() - samp2[col].mean()
        sample_diffs.append(sample_diff)
        if sample_diff > mean_diff:
            counter += 1
            
    alpha = 0.05 
    p=  (counter/10000)     
    print(f"P-value: {p}, is derived from 10,000 Monte Carlo simulations")  #,  Rounded P-value: {np.round((p), 4)}")
    if p <= alpha:
        print(f"Test Conclusion: __Reject H0__          \n")
    else:
        print(f"Test Conclusion: __Fail to reject H0__  \n")
    
    plt.hist(sample_diffs)
    plt.axvline(mean_diff,color = 'k', label="Mean")
    plt.legend()
    plt.title(f"p-value: {counter/10000} | mean value: {np.round(mean_diff,0)}")
    plt.show()
    
    
def shapiro_test(*argv):  
    """ Statistical Normality Tests: Using Shapiro-Wilk Test """
    print(f"Statistical Normality Tests: Using Shapiro-Wilk Test")
    print(f"_____")
    
    alpha = 0.05
    for arg in argv: 
        stat, p = stats.shapiro(arg)
        
        print(f"Statistic: {round(stat, 4)}, P-value: {p},  Rounded P-value: {np.round((p), 4)}")
        if p <= alpha:
            print(f"Test Conclusion: __Reject H0__          Sample does NOT look Gaussian (non-normal distribution)\n")
        else:
            print(f"Test Conclusion: __Fail to reject H0__  Sample looks Gaussian (normal distribution)\n")


def anderson_test(*argv):
    """Statistical Normality Tests: Using Anderson-Darling Test"""
    
    alpha = 0.05
    print(f"Statistical Normality Tests: Using Anderson-Darling Test")
    print(f"_____")
    for arg in argv:
        result = anderson(arg)
        print('\nStatistic: %.3f' % result.statistic)
        p = 0
        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            if result.statistic < result.critical_values[i]:
                print('%.3f: %.3f __Fail to reject H0__   Data looks normal' % (sl, cv))
            else:
                print('%.3f: %.3f __Reject H0__   Data does NOT look normal' % (sl, cv))

def ks_test(sample1, dist):  
    """ Statistical Normality Tests: Using K-S Test """
    print(f"Statistical Normality Tests: Using K-S Test")
    print(f"_____")
    
    alpha = 0.05
    stat, p = stats.kstest(sample1, dist)
        
    print(f"Statistic: {round(stat, 4)}, P-value: {p},  Rounded P-value: {np.round((p), 4)}")
    if p <= alpha:
        print(f"Test Conclusion: __Reject H0__          Sample distribution is NOT identical to a normal distribution\n")
    else:
        print(f"Test Conclusion: __Fail to reject H0__  Sample distribution is identical to a normal distribution")   
        
        
def levene_test(sample1, sample2):
    """ Statistical Variance Tests: Using Levene Variance Test """
    alpha = 0.05
    stat, p = stats.levene(sample1, sample2)
    print(f"Statistic: {round(stat, 4)}, P-value: {p},  Rounded P-value: {np.round((p), 4)}")
    if p <= alpha:
        print(f"Test Conclusion: __Reject H0__     Variances are NOT equal\n")
    else:
        print(f"Test Conclusion: __Fail to reject H0__     Variances are equal (homoscedasticity)")
        

def cohens_d(group1, group2):
    """ Running effect size calculation """
    numer = group1.mean() - group2.mean()
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    denom = np.sqrt(pooled_var)
    return numer / denom


def welch_t(a, b):
    
    """ Calculate Welch's t statistic for two samples. """

    numerator = a.mean() - b.mean()
    
    # “ddof = Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof, 
    #  where N represents the number of elements. By default ddof is zero.
    
    denominator = np.sqrt(a.var(ddof=1)/a.size + b.var(ddof=1)/b.size)
    
    return np.abs(numerator/denominator)

def welch_df(a, b):
    
    """ Calculate the effective degrees of freedom for two samples. This function returns the degrees of freedom """
    
    s1 = a.var(ddof=1) 
    s2 = b.var(ddof=1)
    n1 = a.size
    n2 = b.size
    
    numerator = (s1/n1 + s2/n2)**2
    denominator = (s1/ n1)**2/(n1 - 1) + (s2/ n2)**2/(n2 - 1)
    
    return numerator/denominator


def p_value_welch_ttest(a, b, two_sided=False):
    """Calculates the p-value for Welch's t-test given two samples.
    By default, the returned p-value is for a one-sided t-test. 
    Set the two-sided parameter to True if you wish to perform a two-sided t-test instead.
    """
    t = welch_t(a, b)
    df = welch_df(a, b)
    
    p = 1-stats.t.cdf(np.abs(t), df)
    
    if two_sided:
        return 2*p
    else:
        return p

    
############################################################    
# # Normality Check?
# # Statistical Normality Tests: Using D’Agostino’s K^2 Test

# from scipy.stats import normaltest

# stat, p = normaltest(bev_EU["OrderTotal"])
# print('Statistics=%.3f, p=%.3f' % (stat, p))
# # interpret
# if p > alpha:
#     print(f"p-value is {np.round((p), 4)}, Fail to reject H0, sample looks Gaussian (normal distribution)")
# else:
#     print(f"p-value is {np.round((p), 4)}, Reject H0, sample does not look Gaussian (non-normal distribution)")