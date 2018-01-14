from sim.DiscretePoissonProcess import DiscretePoissonProcess
import matplotlib.pyplot as plt

process = DiscretePoissonProcess(2.5)
nums = [process.get_discrete_increase() for x in range(50)]

print("The number of events is: ", sum(nums))
fig = plt.figure()
plt.plot(nums)
plt.show()