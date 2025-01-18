import matplotlib.pyplot as plt
import numpy as np

numcourses = [13,4,12,3,14,13,12,9,11,7,13,11,9,2,5,7,10,0,9,7]
happiness = [70,25,54,21,80,68,84,62,57,40,60,64,45,38,51,52,58,21,75,70]
plt.figure(figsize=(6,6))

plt.plot(numcourses,happiness,'ks',markersize=15)
plt.xlabel('Number of courses taken')
plt.ylabel('General life happiness')
plt.xlim([-1,15])
plt.ylim([0,100])
plt.grid()
plt.xticks(range(0,15,2))
plt.savefig('Figure_11_03.png',dpi=300)
plt.show()
# Build a statistical model
# design matrix as a column vector
X = np.array(numcourses,ndmin=2).T
print(X.shape)
# fit the model using the left-inverse
X_leftinv = np.linalg.inv(X.T@X) @ X.T
# solve for the coefficients
beta = X_leftinv @ happiness
beta
# let's plot it!
# predicted data
pred_happiness = X@beta
plt.figure(figsize=(6,6))
# plot the data and predicted values
plt.plot(numcourses,happiness,'ks',markersize=15)
plt.plot(numcourses,pred_happiness,'o-',color=[.6,.6,.6],linewidth=3,markersize=8)
# plot the residuals (errors)
for n,y,yHat in zip(numcourses,happiness,pred_happiness):
    plt.plot([n,n],[y,yHat],'--',color=[.8,.8,.8],zorder=-10)
plt.xlabel('Number of courses taken')
plt.ylabel('General life happiness')
plt.xlim([-1,15])
plt.ylim([0,100])
plt.xticks(range(0,15,2))
plt.legend(['Real data','Predicted data','Residual'])
plt.title(f'SSE = {np.sum((pred_happiness-happiness)**2):.2f}')
plt.savefig('Figure_11_04.png',dpi=300)
plt.show()