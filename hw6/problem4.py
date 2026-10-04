import numpy as np
import matplotlib.pyplot as plt



def convolution(x1, x2):    
    # Find length of the output x 
    output_length = x1.shape[0] + x2.shape[0] - 1
    output = np.zeros(output_length)  # Initialize the output array with zeros

    # Perform the convolution
    for i in range(output_length):
        for j in range(x2.shape[0]):
            if (i - j) >= 0 and (i - j) < x1.shape[0]:
                output[i] = # insert code here

    return output


### Problem 4.1

l = 3
lconv = 2*l - 1
x1 = np.array([1.,-1.,1.])
x2 = np.array([2.,-7.,1.])

z1 = convolution(x1, x2)
z2 = np.convolve(x1, x2)
print("For problem 4.1, difference between the result from numpy and code is = ",z1 - z2)


f1, axs = plt.subplots(3,1)
ax = axs[0]
ax.stem(np.arange(l),x1)
ax.set_ylabel("x1")
ax.set_xticks(np.arange(l))

ax = axs[1]
ax.stem(np.arange(l),x2)
ax.set_ylabel("x2")
ax.set_xticks(np.arange(l))

ax = axs[2]
ax.stem(np.arange(lconv),z1)
ax.set_ylabel("z1")
ax.set_xticks(np.arange(lconv))

f1.suptitle("Problem 4.1")
plt.show()
plt.close()

### Problem 4.2

trunc = 5
l = 2*trunc + 1
lconv = 2*l - 1

x1 = np.concatenate((np.zeros(5),np.ones(6)),axis=0)
x2 = np.concatenate((np.zeros(3),np.ones(5),np.zeros(3)),axis=0)

z1 = convolution(x1, x2)
z2 = np.convolve(x1, x2)
print("For problem 4.2, difference between the result from numpy and code is = ",z1 - z2)

f1, axs = plt.subplots(3,1)
ax = axs[0]
ax.stem(np.arange(l),x1)
ax.set_ylabel("x1")
tick_labels = [str(x) for x in np.arange(-trunc,trunc+1)]
ax.set_xticks(np.arange(l))
ax.set_xticklabels(tick_labels)

ax = axs[1]
ax.stem(np.arange(l),x2)
ax.set_ylabel("x2")
ax.set_xticks(np.arange(l))
ax.set_xticklabels(tick_labels)

ax = axs[2]
ax.stem(np.arange(lconv)-trunc*2,z1)
ax.set_ylabel("z1")
ax.set_xticks(np.arange(lconv)-trunc*2)

f1.suptitle("Problem 4.2")
plt.show()
plt.close()
