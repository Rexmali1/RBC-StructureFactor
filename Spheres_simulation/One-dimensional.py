%matplotlib widget
import numpy as np
import sys
from matplotlib import pyplot as plt
import matplotlib.animation as animation

def collisions(P,V,N,r):
  for j in range(N):
    for k in range(j):
      if np.linalg.norm(P[j,:]-P[k,:]) <= 2*r:
        P21    = P[j,:]-P[k,:]
        P21    = P21 / np.linalg.norm(P21)
        PM     = (1/2)*(P[k,:]+P[j,:])
        P[j,:] = PM + r*P21
        P[k,:] = PM - r*P21
        
        V1     = V[k,:] - (np.dot((V[k,:]-V[j,:]), P21))*P21
        V2     = V[j,:] + (np.dot((V[k,:]-V[j,:]), P21))*P21
        V[k,:] = V1        
        V[j,:] = V2
      else:
        pass
        
def wall(P,V,N,r,L):
  for j in range(N):
    if P[j][0] <= r and V[j][0] < 0:
      V[j][0] = -V[j][0]
    elif P[j][0] >= L-r and V[j][0] > 0:
      V[j][0] = -V[j][0]
    else:
      pass


#Parameters of simulation
N    = 9           #Number of spheres
L    = 10.0        #Length of box
r    = 0.5         #Radius of spheres 
h    = 0.001       #Intervale
tsim = 10          #Time of simulation
ite  = int(tsim/h) #Number of iterations
t    = 0           #Initial time

#Speed of particles
V  = np.transpose(np.array([10*(np.random.rand(N)-0.5),np.zeros(N)]))   

#Initial position
if N < L/(2*r):
  P   = np.linspace(r, L-r, num=N) 
  P   = np.transpose(np.array([P,np.ones(N)*r]))
  p_s = np.zeros([int(ite/42 +2),N,2])
else:
  print("Intruduce un numero de esferas menor")
  exit()
    
#Create a figure to plot
fig, ax = plt.subplots(figsize=(10,2))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Simulation of HS in 1D')
ax.set_xlim([0, L])
ax.set_ylim([0, 2])
scatter = ax.scatter(P[:,0],P[:,1],s=53**2)

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
  scatter.set_offsets(p_s[i,:,:]) 
  return scatter,

                
    
ani = animation.FuncAnimation(fig, update, frames=range(int(ite/42)), 
                              blit=True, interval=42,repeat=False)
ani.save("Simulation-1D.gif")
plt.show()
