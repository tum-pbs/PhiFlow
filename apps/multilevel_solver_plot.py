import numpy as np
import pylab

mg = np.loadtxt("C:/Users/Philipp/model/simpleplume128x128/sim_000030/iter.txt")
cg = np.loadtxt("C:/Users/Philipp/model/simpleplume128x128/sim_000028/iter.txt")

pylab.plot(mg[:,0], label="MG low-res")
pylab.plot(mg[:,1], label="MG high-res")
pylab.plot(cg, label="CG")
pylab.legend()
pylab.show()