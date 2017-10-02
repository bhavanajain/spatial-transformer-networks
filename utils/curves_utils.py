import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

acc_regex = r"Accuracy \((\d+)\): ((\d+)\.(\d+))"
loss_regex = r"Iteration: 90 Loss: ((\d+)\.(\d+))"

acc = []
loss = []

with open("logs.txt", "r") as f:
    for line in f:
        lr = re.match(loss_regex, line)
        if lr:
            loss.append(lr.group(1))
            continue    
        ar = re.match(acc_regex, line)
        if ar:
            acc.append(ar.group(2))
            continue

plt.plot(range(0, len(acc)), acc)
plt.xlabel("Epoch number")
plt.ylabel("Accuracy")
plt.title("Accuracy plot for tensorflow github model")
plt.savefig('accuracy.png')
plt.close()


plt.plot(range(0, len(loss)), loss)
plt.xlabel("Epoch number")
plt.ylabel("Losses")
plt.title("Plotting losses for tensorflow github model")
plt.savefig('loss.png')
plt.close()
