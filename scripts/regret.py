import matplotlib.pyplot as plt
import sys

input = sys.argv[1]
output = sys.argv[2]

x = []
y = []
for line in open(input).readlines():
    nums = [float(f) for f in line.split(",")]
    x.append(nums[0])
    y.append(nums[1])

plt.plot(x, y)
plt.title("Regret")
plt.xlabel("trials")
plt.ylabel("regret")

plt.savefig(output)
