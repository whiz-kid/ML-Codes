# Import 
import matplotlib.pyplot as plt

# Initialize the plot
fig = plt.figure()
ax=fig.add_subplot(221)
x=[1,2,3,4]
y=[4,5,9,7]
ax.plot(x,y)

#ax1 = fig.add_subplot(121)
#ax2 = fig.add_subplot(122)

# Show the plot
plt.show()