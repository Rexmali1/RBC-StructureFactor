%matplotlib widget
import numpy as np
import sys
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d

def collisions(P,V,N,r):
  for j in range(N):
    for k in range(j):
      if np.linalg.norm(P[j,:]-P[k,:]) <= 2*r:
        #Assign speeds after collisio
        P21    = P[j,:]-P[k,:]
        P21    = P21 / np.linalg.norm(P21)
        PM     = (1/2)*(P[k,:]+P[j,:])
        P[j,:] = PM + r*P21
        P[k,:] = PM - r*P21
        
        #Assign speeds after collision 
        V1     = V[k,:] - np.dot(V[k,:]-V[j,:], P21) * P21
        V2     = V[j,:] + np.dot(V[k,:]-V[j,:], P21) * P21
        V[k,:] = V1        
        V[j,:] = V2
      else:
        pass
        
def wall(P,V,N,r,L):
  for j in range(N):
    #Collision with x walls
    if P[j,0] <= r:
      P[j,0] = r
      V[j,0] = -V[j,0]
    elif P[j,0] >= L-r:
      P[j,0] = L-r
      V[j,0] = -V[j,0]
    else: 
      pass
    #Collision with y walls
    if P[j,1] <= r:
      P[j,1] = r
      V[j,1] = -V[j,1]
    elif P[j,1] >= L-r:
      P[j,1] = L-r
      V[j,1] = -V[j,1]
    else:
      pass
    #Collision with z walls
    if P[j,2] <= r:
      P[j,2] = r
      V[j,2] = -V[j,2]
    elif P[j,2] >= L-r:
      P[j,2] = L-r
      V[j,2] = -V[j,2]
    else:
      pass



#Parameters of simulation
N    = 300         #Number of spheres
L    = 10.0        #Length of box
r    = 0.5         #Radius of spheres 
h    = 0.001       #Intervale
tsim = 10          #Time of simulation
ite  = int(tsim/h) #Number of iterations
t    = 0           #Initial time

#Speed of particles
V  = 10*(np.random.rand(N,3)-0.5) 

#Initial position

P   = ((L-2*r)*np.random.rand(N,3))+r
p_s = np.zeros([int(ite/42 +1),N,3])

    
#Create a figure to plot
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Simulation of HS in 3D')
ax.set_xlim([0, L])
ax.set_ylim([0, L])
ax.set_zlim([0, L])
scatter = ax.scatter(P[:,0],P[:,1],P[:,2],s=30**2)

a=0
for i in range(ite):
  #Save frames to animation
  if i%42 == 0:
    p_s[a,:,:] = P
    a          = a + 1
  # Motion
  P = P + h*V
  t = t + h
  collisions(P,V,N,r)
  wall(P,V,N,r,L) 
  

def update(i):
  scatter._offsets3d = (p_s[i,:,0], p_s[i,:,1], p_s[i,:,2])
  return scatter,

                
    
ani = animation.FuncAnimation(fig, update, frames=range(int(ite/42)), 
                              blit=True, interval=42,repeat=False)
ani.save("Simulation-3D.gif")
plt.show()
