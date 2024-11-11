import numpy as np

def one_way_anova(*args):
    # Number of groups
    k = len(args)
    # Total number of observations
    N = sum(len(group) for group in args)
    # Group means
    group_means = [np.mean(group) for group in args]
    # Overall mean
    grand_mean = np.mean([item for group in args for item in group])

    # Between-group sum of squares (SSB)
    SSB = sum(len(group) * ((group_mean - grand_mean) ** 2) for group, group_mean in zip(args, group_means))

    # Within-group sum of squares (SSW)
    SSW = sum(sum((item - group_mean) ** 2 for item in group) for group, group_mean in zip(args, group_means))

    # Between-group degrees of freedom (dfB)
    dfB = k - 1
    # Within-group degrees of freedom (dfW)
    dfW = N - k

    # Mean square between (MSB)
    MSB = SSB / dfB
    # Mean square within (MSW)
    MSW = SSW / dfW

    # F statistic
    F = MSB / MSW

    return F, dfB, dfW

# Example usage:
# Data for three groups
group1 = [6, 8, 4, 5, 3, 4]
group2 = [8, 12, 9, 11, 6, 8]
group3 = [13, 9, 11, 8, 7, 12]

# Perform one-way ANOVA
F_statistic, df_between, df_within = one_way_anova(group1, group2, group3)
print("\nOne Way ANOVA\n")
print(f"F-statistic: {F_statistic}")
print(f"Degrees of freedom between groups: {df_between}")
print(f"Degrees of freedom within groups: {df_within}")

# Function to calculate two-way ANOVA
def two_way_anova(data, alpha=0.05):
    # Number of levels for each factor
    a = len(data)
    b = len(data[0])
    # Total number of observations
    n = len(data[0][0])
    N = a * b * n

    # Calculate sum of squares
    SS_total = sum(((x - np.mean(data)) ** 2) for group in data for subgroup in group for x in subgroup)
    SS_factor_A = b * n * sum((np.mean(group) - np.mean(data)) ** 2 for group in data)
    SS_factor_B = a * n * sum((np.mean(subgroup) - np.mean(data)) ** 2 for group in data for subgroup in group)
    SS_error = SS_total - SS_factor_A - SS_factor_B

    # Calculate degrees of freedom
    df_total = N - 1
    df_factor_A = a - 1
    df_factor_B = b - 1
    df_error = df_total - df_factor_A - df_factor_B

    # Calculate mean squares
    MS_factor_A = SS_factor_A / df_factor_A
    MS_factor_B = SS_factor_B / df_factor_B
    MS_error = SS_error / df_error

    # Calculate F-statistics
    F_factor_A = MS_factor_A / MS_error
    F_factor_B = MS_factor_B / MS_error

    return {
        'SS_total': SS_total,
        'SS_factor_A': SS_factor_A,
        'SS_factor_B': SS_factor_B,
        'SS_error': SS_error,
        'df_total': df_total,
        'df_factor_A': df_factor_A,
        'df_factor_B': df_factor_B,
        'df_error': df_error,
        'MS_factor_A': MS_factor_A,
        'MS_factor_B': MS_factor_B,
        'MS_error': MS_error,
        'F_factor_A': F_factor_A,
        'F_factor_B': F_factor_B
    }

data = [
    [[3, 2, 1], [4, 5, 6]],
    [[5, 6, 7], [8, 9, 10]],
    [[7, 8, 9], [10, 11, 12]]
]

print("\nTwo Way ANOVA\n")
# Perform two-way ANOVA
results = two_way_anova(data)

for key, value in results.items():
    print(f"{key}: {value}")
