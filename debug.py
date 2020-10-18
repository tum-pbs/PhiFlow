from phi.flow import *

a = math.random_normal(batch=10, channels=2, x=64, y=32)

a = math.laplace(a, dx=[2, 2])
# b = math.gradient(a)

print(a)