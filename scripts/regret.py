import matplotlib.pyplot as plt
import sys

input = sys.argv[1]
output = sys.argv[2]

regret = 0.
data = []
for line in open(input).readlines():
    data.append(regret)
    regret += float(line)

plt.plot(data)
plt.title("Regret")
plt.xlabel("trials")
plt.ylabel("regret")

plt.savefig(output)
