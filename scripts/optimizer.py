import matplotlib.pyplot as plt
import sys

input = sys.argv[1]
output = sys.argv[2]

x = []
y = []
z = []
for line in open(input).readlines():
    nums = [float(f) for f in line.split(",")]
    x.append(nums[0])
    y.append(nums[1])
    z.append(nums[2])

plt.scatter(x, y, c=z, s = 20)
plt.title("Optimization output")
plt.xlabel("alpha")
plt.ylabel("beta")

plt.savefig(output)
