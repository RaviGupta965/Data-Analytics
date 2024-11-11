# Import the necessary libraries
import numpy as np
import scipy.stats as stats

print("\nOne Tailed Test\n")

print("A school claimed that the students who study there are more intelligent than the average school. On calculating the IQ scores of 50 students, the average turns out to be 110. The mean of the population IQ is 100 and the standard deviation is 15. State whether the claim of the principal is right or not at a 5% significance level.")

# Given information
sample_mean = 110
population_mean = 100
population_std = 15
sample_size = 50
alpha = 0.05

print("Sample Mean:", sample_mean)
print("Population Mean:", population_mean)
print("Population standard deviation:", population_std)
print("Sample size:", sample_size)
print("Level of significance:", alpha)

# Compute the z-score
z_score = (sample_mean - population_mean) / (population_std / np.sqrt(sample_size))
print('Z-Score:', z_score)

print("\nUsing Critical Z-Score\n")

# Critical Z-Score
z_critical = stats.norm.ppf(1 - alpha)
print('Critical Z-Score:', z_critical)

# Hypothesis
if z_score > z_critical:
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")

print("\nApproach 2: Using P-value\n")

# P-Value : Probability of getting less than a Z-score
p_value = 1 - stats.norm.cdf(z_score)

print('p-value:', p_value)

# Hypothesis
if p_value < alpha:
    print("Reject Null Hypothesis")
else:
    print("Fail to Reject Null Hypothesis")

print("\nTwo-sampled z-test:\n")
print("There are two groups of students preparing for a competition: Group A and Group B. Group A has studied offline classes, while Group B has studied online classes. After the examination, the score of each student comes. Now we want to determine whether the online or offline classes are better.")

print("Group A: Sample size = 50, Sample mean = 75, Sample standard deviation = 10")
print("Group B: Sample size = 60, Sample mean = 80, Sample standard deviation = 12")
print("Assuming a 5% significance level, perform a two-sample z-test to determine if there is a significant difference between the online and offline classes.")

# Group A (Offline Classes)
n1 = 50
x1 = 75
s1 = 10

# Group B (Online Classes)
n2 = 60
x2 = 80
s2 = 12

# Null Hypothesis = mu_1 - mu_2 = 0
# Hypothesized difference (under the null hypothesis)
D = 0

# Set the significance level
alpha = 0.05

# Calculate the test statistic (z-score)
z_score = ((x1 - x2) - D) / np.sqrt((s1**2 / n1) + (s2**2 / n2))
print('Z-Score:', np.abs(z_score))

# Calculate the critical value
z_critical = stats.norm.ppf(1 - alpha / 2)
print('Critical Z-Score:', z_critical)

# Compare the test statistic with the critical value
if np.abs(z_score) > z_critical:
    print("Reject the null hypothesis. There is a significant difference between the online and offline classes.")
else:
    print("Fail to reject the null hypothesis. There is not enough evidence to suggest a significant difference between the online and offline classes.")

# Approach 2: Using P-value
p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))
print('P-Value:', p_value)

# Compare the p-value with the significance level
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference between the online and offline classes.")
else:
    print("Fail to reject the null hypothesis. There is not enough evidence to suggest a significant difference between the online and offline classes.")

print("\nIndependent samples t-test:\n")

# Sample
sample_A = np.array([1, 2, 4, 4, 5, 5, 6, 7, 8, 8])
sample_B = np.array([1, 2, 2, 3, 3, 4, 5, 6, 7, 7])

# Perform independent sample t-test
t_statistic, p_value = stats.ttest_ind(sample_A, sample_B)

# Set the significance level (alpha)
alpha = 0.05

# Compute the degrees of freedom (df) (n_A-1) + (n_B-1)
df = len(sample_A) + len(sample_B) - 2

# Calculate the critical t-value
critical_t = stats.t.ppf(1 - alpha / 2, df)

# Print the results
print("T-value:", t_statistic)
print("P-Value:", p_value)
print("Critical t-value:", critical_t)

# Decision
print('With T-value')
if np.abs(t_statistic) > critical_t:
    print('There is significant difference between two groups')
else:
    print('No significant difference found between two groups')

print('With P-value')
if p_value > alpha:
    print('No evidence to reject the null hypothesis that a significant difference exists between the two groups')
else:
    print('Evidence found to reject the null hypothesis that a significant difference exists between the two groups')

print("\nPaired sample t-test\n")

# Create the paired samples
math1 = np.array([4, 4, 7, 16, 20, 11, 13, 9, 11, 15])
math2 = np.array([15, 16, 14, 14, 22, 22, 23, 18, 18, 19])

# Perform the paired sample t-test
t_statistic, p_value = stats.ttest_rel(math1, math2)

# Compute the degrees of freedom (df=n-1)
df = len(math2) - 1

# Calculate the critical t-value
critical_t = stats.t.ppf(1 - alpha / 2, df)

# Print the results
print("T-value:", t_statistic)
print("P-Value:", p_value)
print("Critical t-value:", critical_t)

# Decision
print('With T-value')
if np.abs(t_statistic) > critical_t:
    print('There is significant difference between math1 and math2')
else:
    print('No significant difference found between math1 and math2')

print('With P-value')
if p_value > alpha:
    print('No evidence to reject the null hypothesis that significant difference exists between math1 and math2')
else:
    print('Evidence found to reject the null hypothesis that significant difference exists between math1 and math2')

print("\nOne sample t-test\n")

# Define the population mean weight
population_mean = 45

# Define the sample mean weight and standard deviation
sample_mean = 75
sample_std = 25

# Define the sample size
sample_size = 25

# Calculate the t-statistic
t_statistic = (sample_mean - population_mean) / (sample_std / np.sqrt(sample_size))

# Define the degrees of freedom
df = sample_size - 1

# Calculate the critical t-value
critical_t = stats.t.ppf(1 - alpha, df)

# Calculate the p-value
p_value = 1 - stats.t.cdf(t_statistic, df)

# Print the results
print("T-Statistic:", t_statistic)
print("Critical t-value:", critical_t)
print("P-Value:", p_value)

# Decision
print('With T-value:')
if t_statistic > critical_t:
    print("There is a significant difference in weight before and after the camp. The fitness camp had an effect.")
else:
    print("There is no significant difference in weight before and after the camp. The fitness camp did not have a significant effect.")

print('With P-value:')
if p_value > alpha:
    print("There is a significant difference in weight before and after the camp. The fitness camp had an effect.")
else:
    print("There is no significant difference in weight before and after the camp. The fitness camp did not have a significant effect.")
