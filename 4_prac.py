import numpy as np
import math
import matplotlib.pyplot as plt
# lam : lambda
def poisson_dist(x,lam):
    u=[]
    z=[]
    for i in range(len(x)):
        u.append(math.factorial(i))
        z.append(lam**i)
        #print(prob_density)
    q=np.zeros(len(x))
    p=np.zeros(len(x))
    q=np.array(u)
    p=np.array(z)
    wz=p/q # lambda^k/k!
    prob_density=wz*np.exp(-lam)
    return prob_density
 
lam=1
x = np.arange(0, 40, 1)
result = poisson_dist(x,lam)

fig, axs = plt.subplots(2, 3)

axs[0, 0].set_title('lam=1')
axs[0, 0].bar(x, result)
axs[0, 0].set_title('lam=1')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('Probability')

lam=2
result = poisson_dist(x,lam)

axs[0, 1].set_title('lam=2')
axs[0, 1].bar(x, result)
axs[0, 1].set_title('lam=2')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('Probability')

lam=5
result = poisson_dist(x,lam)

axs[0, 2].set_title('lam=5')
axs[0, 2].bar(x,result)
axs[0, 2].set_xlabel('x')
axs[0, 2].set_ylabel('Probability')

lam=10
result = poisson_dist(x,lam)

axs[1, 0].set_title('lam=10')
axs[1, 0].bar(x, result)
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('Probability')

lam=15
result = poisson_dist(x,lam)
axs[1, 1].set_title('lam=15')
axs[1, 1].bar(x, result)
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('Probability')

lam=20
result = poisson_dist(x,lam)

axs[1, 2].set_title('lam=20')
axs[1, 2].bar(x, result)
axs[1, 2].set_xlabel('x')
axs[1, 2].set_ylabel('Probability')

fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.6, wspace=0.8)
plt.show()
