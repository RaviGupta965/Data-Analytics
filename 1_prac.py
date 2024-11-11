import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('UnemploymentRates.csv', names=['Column A'])
values = data['Column A'].tolist()
values.sort()


hist = plt.hist(values, bins=[1, 3, 5, 7, 9, 11, 13],
                edgecolor='black', density=True)
print(hist)


plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Data visualization using Histogram')
plt.show()