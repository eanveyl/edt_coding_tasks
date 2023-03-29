import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import rayleigh


## A2 d)

# Case a)
mean = 0
variance = 2
std_dev = np.sqrt(variance)
Y = np.random.normal(loc=mean, scale=std_dev, size=10000)

plt.hist(Y, bins=500, density=True)
x = np.linspace(-5, 5, 100)
plt.plot(x, norm.pdf(x, mean, std_dev))  # theoretical distribution is Normal distribution

plt.xlabel('Y')
plt.ylabel('PDF')
plt.title("A2 Case a)")
plt.legend()
plt.show()


# Case b) 
X1 = np.random.normal(loc=mean, scale=std_dev, size=10000)
X2 = np.random.normal(loc=mean, scale=std_dev, size=10000)
R = np.sqrt(X1**2 + X2**2)

plt.hist(R, bins=500, density=True)
x = np.linspace(-5, 5, 100)
#plt.plot(x, norm.pdf(x, mean, std_dev))
plt.plot(x, rayleigh.pdf(x, mean, std_dev))  # theoretical distribution is Rayleigh distribution

plt.xlabel('Y')
plt.ylabel('PDF')
plt.title("A2 Case b)")
plt.legend()
plt.show()

# Case c) TODO: not finished
Z = (R>1).astype(int)
plt.hist(Z, bins=500, density=True)
plt.xlabel('Y')
plt.ylabel('PDF')
plt.title("A2 Case c)")
plt.show()


## A3

N = 3
mean = 0
variance = 2
std_dev = np.sqrt(variance)

# a)
X = np.random.normal(loc=mean, scale=std_dev, size=10000)
for n in range(N-1):
    X_temp = np.random.normal(loc=mean, scale=std_dev, size=10000)
    X = np.vstack((X, X_temp))

Y = np.sum(X, axis=0)

plt.hist(Y, bins=500, density=True)
x = np.linspace(-20, 20, 1000)
plt.plot(x, norm.pdf(x, N*mean, np.sqrt(N)*std_dev))
plt.xlabel('Y')
plt.ylabel('PDF')
plt.title("A3 a)")
plt.show()

# b)
X = np.random.uniform(-5, 5, size=10000)
for n in range(N-1):
    X_temp = np.random.uniform(-5, 5, size=10000)
    X = np.vstack((X, X_temp))
Y = np.sum(X, axis=0)

std_dev = np.sqrt((1/12)*(5--5)**2)
mean = 0.5*(-5+5)

plt.hist(Y, bins=500, density=True)
x = np.linspace(-20, 20, 1000)
plt.plot(x, norm.pdf(x, N*mean, np.sqrt(N)*std_dev))
plt.xlabel('Y')
plt.ylabel('PDF')
plt.title("A3 b)")
plt.show()

# c) Same thing again but for Rayleigh distribution


## A4

# PARAMETERS
# number of points to generate
N_smp = 500

# bi-variate Normal distribution
p = 1#0.8         # correlation coefficient
mean = [0, 0]   # mean vector
cov = [[1, p],  # covariance matrix
       [p, 1]]

# line parameters
theta = np.pi/6 # angle relative to the x-axis
b = 0           # y-axis intercept (offset)

# RANDOM POINTS and LINE
# NOTE: Fill-in the missing code

# generate points
X = np.random.multivariate_normal(mean, cov, N_smp)

# calculate probability of a point being on one side of the line
def is_above(x, y, m, b=0):
    return y > m*x + b

count = 0
m = np.tan(theta)
for i in range(N_smp):
    x, y = X[i]
    if is_above(x, y, m, b):
        count += 1
p = count / N_smp  # empirical!

# PRINT > PROBABILITIES
print('Probabilities of point being on each side of the line:')
print('   p1 = {:0.3f} \t p2 = {:0.3f}'.format(p, 1-p))

# PLOT > POINTS and LINE
# useful points to draw the line
l = np.array([np.cos(theta),  # line parallel vector
              np.sin(theta)])
x1 = np.array([0, b]) - 20*l
x2 = np.array([0, b]) + 20*l

# plot
print(X[0:10])
plt.plot(X[0], X[1], '.', label='points')
plt.plot([x1[0], x2[0]], [x1[1], x2[1]], label='line')
plt.gca().set_aspect('equal', 'box')
plt.grid()
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Bi-variate Normal distribution")
plt.show()

# push test R&S
# push test 2