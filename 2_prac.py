import numpy as np
import matplotlib.pyplot as plt


# Generating random data in a controlled fashion.
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
outlier_low = np.random.rand(10) * -100
outlier_high = np.random.rand(10) * 100 + 100


data = np.concatenate((spread, center, outlier_low, outlier_high))


# Displaying box plots on subplots with varying settings.
fig, axs = plt.subplots(2, 3)


axs[0, 0].boxplot(data)
axs[0, 0].set_title('Basic Plot')


axs[0, 1].boxplot(data, 1)
axs[0, 1].set_title('Notched Plot')


axs[0, 2].boxplot(data, 0, 'gD')
axs[0, 2].set_title('Change Outlier\nPoint Symbols')


axs[1, 0].boxplot(data, 0, '')
axs[1, 0].set_title("Don't Show\nOutlier Points")


axs[1, 1].boxplot(data, 0, 'rs', 0)
axs[1, 1].set_title('Horizontal Boxes')


axs[1, 2].boxplot(data, 0, 'rs', 0, 0.75)
axs[1, 2].set_title('Change Wisker Length')


fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1,
                    hspace=0.5, wspace=0.5)
plt.show()