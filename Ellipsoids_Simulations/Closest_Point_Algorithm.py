import numpy as np
from random import random

#Number of ellipsoids
n = 2 

#Ellipsoids parameters
a = 4.0
b = 4.0
c = 1.5

D = np.array([[1/(a**2),    0   ,    0   ],
              [   0    ,1/(b**2),    0   ],
              [   0    ,    0   ,1/(c**2)]])

GRADIENT_ZERO_TOLERANCE = 10**(-6)
HESSIAN_ZERO_TOLERANCE  = 10**(-12)
EV_ZERO_TOLERANCE       = 10**(-6)
NEWTON_ITERATION_LIMIT  = 100


def x(alpha,beta):  #Give the coordinates of a point in ellipsoid surface
  x_cord = a * np.cos(alpha)*np.cos(beta)
  y_cord = b * np.cos(alpha)*np.sin(beta)
  z_cord = c * np.sin(alpha)
  return np.array([x_cord,y_cord,z_cord])

def x_alpha(alpha,beta):  #Give the partial derivative with respect to alpha of a point in ellipsoid surface
  x_cord = -a * np.sin(alpha)*np.cos(beta)
  y_cord = -b * np.sin(alpha)*np.sin(beta)
  z_cord =  c * np.cos(alpha)
  return np.array([x_cord,y_cord,z_cord])

def x_beta(alpha,beta):  #Give the partial derivative with respect to beta of a point in ellipsoid surface
  x_cord = -a * np.cos(alpha)*np.sin(beta)
  y_cord =  b * np.cos(alpha)*np.cos(beta)
  z_cord = 0
  return np.array([x_cord,y_cord,z_cord])

def x_alpha_alpha(alpha,beta):  #Give the second partial derivative with respect to alpha of a point in ellipsoid surface
  return -x(alpha,beta)

def x_beta_beta(alpha,beta):   #Give the second partial derivative with respect to beta of a point in ellipsoid surface
  x_cord = -a * np.cos(alpha)*np.cos(beta)
  y_cord = -b * np.cos(alpha)*np.sin(beta)
  z_cord = 0
  return np.array([x_cord,y_cord,z_cord])

def x_alpha_beta(alpha,beta):  #Give the second cross partial derivative of a point in ellipsoid surface
  x_cord =  a * np.sin(alpha)*np.sin(beta)
  y_cord = -b * np.sin(alpha)*np.cos(beta)
  z_cord = 0
  return np.array([x_cord,y_cord,z_cord])


def rotation(alpha,beta,gamma):  #Generate a rotation matrix
  m_x = np.array([[1 ,       0      ,       0       ],
                  [0 , np.cos(alpha), -np.sin(alpha)],
                  [0 , np.sin(alpha),  np.cos(alpha)]])
  
  m_y = np.array([[ np.cos(beta), 0 , np.sin(beta)],
                  [      0      , 1 ,      0      ],
                  [-np.sin(beta), 0 , np.cos(beta)]])
  
  m_z = np.array([[np.cos(gamma),-np.sin(gamma), 0],
                  [np.sin(gamma), np.cos(gamma), 0],
                  [      0      ,        0     , 1]])
  
  return np.matmul(m_z , np.matmul(m_y, m_x) )

def margin(theta, K, c21): #Returns the margin function that quantifies the distance between two ellipsoids
  return np.dot(x(theta[0], theta[1])- c21 , np.matmul(K, x(theta[0], theta[1])- c21) )
  

def ComputeGradient(theta, c21, K):  #Calculate the gradient of the margin function in a point of a ellipsoid surface
  gradient    = np.zeros(2)
  gradient[0] = 2 *( np.dot( (x(theta[0],theta[1]) - c21) , np.matmul(K , x_alpha(theta[0],theta[1])) ))
  gradient[1] = 2 *( np.dot( (x(theta[0],theta[1]) - c21) , np.matmul(K , x_beta(theta[0],theta[1])) ))
  return gradient


def ComputeHessian(theta, c21, K): #Calculate the Hesian of the margin function in a point of a ellipsoid surface
  hessian      = np.zeros([2,2])
  hessian[0,0] =2*(np.dot( x_alpha(theta[0],theta[1]), np.matmul( K , x_alpha(theta[0],theta[1]))) + np.dot( x(theta[0],theta[1]) - c21 , np.matmul(K , x_alpha_alpha(theta[0],theta[1]))))
  hessian[0,1] =2*(np.dot( x_beta(theta[0],theta[1]) , np.matmul( K , x_alpha(theta[0],theta[1]))) + np.dot( x(theta[0],theta[1]) - c21 , np.matmul(K , x_alpha_beta(theta[0],theta[1]))))
  hessian[1,0] =2*(np.dot( x_alpha(theta[0],theta[1]), np.matmul( K , x_beta(theta[0],theta[1]) )) + np.dot( x(theta[0],theta[1]) - c21 , np.matmul(K , x_alpha_beta(theta[0],theta[1]))))
  hessian[1,1] =2*(np.dot( x_beta(theta[0],theta[1]) , np.matmul( K , x_beta(theta[0],theta[1]) )) + np.dot( x(theta[0],theta[1]) - c21 , np.matmul(K , x_beta_beta(theta[0],theta[1]))))
  return hessian

def ComputeCorrectedHessian(hessian):  #This function prevents the Hessian causing divergences
  ev = np.linalg.eigvals(hessian)
  for i in range(2):
    if ev[i] < 0:
      ev[i] = - ev[i]
    if ev[i] < EV_ZERO_TOLERANCE:
      ev[i] = 1

  hessian[0,0] = ev[0]
  hessian[0,1] = 0
  hessian[1,0] = 0
  hessian[1,1] = ev[1]

  return hessian

def computeStepSize(dir, tau, gamma, sigma ,c21, grad ,K, theta): #Calculate a good Step Size for the application of the Newton method
  var = gamma

  f1 = margin(theta, K, c21)
  f2 = margin(theta + var*dir, K, c21)

  while f2 > f1 + var * sigma * np.dot(grad,dir) :
    var *= tau
    f2 =margin(theta+ var*dir, K, c21)

  return var*dir

#Angles of the representative point
E_rep =[[0,0],[0,np.pi],[np.pi/2,0],[-np.pi/2,0],[0,np.pi/2],[0,-np.pi/2],  #Final points
              
        [np.pi/4,np.pi/4],[np.pi/4,-np.pi/4],[np.pi/4,3*np.pi/4],[np.pi/4,-3*np.pi/4],
        [-np.pi/4,np.pi/4],[-np.pi/4,-np.pi/4],[-np.pi/4,3*np.pi/4],[-np.pi/4,-3*np.pi/4],  #Octant points

        [-np.pi/4,-np.pi/2],[-np.pi/4,0],[-np.pi/4,np.pi/2],[-np.pi/4,np.pi],
        [0,-3*np.pi/4],[0,-np.pi/4],[0,np.pi/4],[0,3*np.pi/4],
        [np.pi/4,-np.pi/2],[np.pi/4,0],[np.pi/4,np.pi/2],[np.pi/4,np.pi]]   #Fill points


#Generate a random position of centroid of ellipsoids
centroids = 5*np.random.rand(n,3) #Centroid of ellipsoids

rotation_M = np.zeros([n,3,3])

#Generate n random rotation matrix
for i in range(n):
  rotation_M[i] = rotation(2*np.pi*random(),2*np.pi*random(),2*np.pi*random())


c21= np.matmul( rotation_M[0].transpose() , centroids[1]-centroids[0])

K = np.matmul( rotation_M[0].transpose() ,  np.matmul( rotation_M[1] , np.matmul( D, np.matmul( rotation_M[1].transpose() , rotation_M[0]))))

f_rep = np.zeros(26)

for i in range(26):
  theta = np.array([E_rep[i][0],E_rep[i][0]])
  f_rep[i] = margin(theta, K, c21)

lowest = f_rep[0]
j=0

for i in range(26):
  if f_rep[i] <= lowest:
    lowest = f_rep[i]
    j=i

if lowest <= 1:
  theta   = np.array([E_rep[j][0], E_rep[j][1]])
  grad    = ComputeGradient(theta,c21,K)
  hessian = ComputeHessian(theta,c21,K)

  iterCount = 0
  tau       = 0.5
  sigma     = 0.01
  gamma     = 1.0

  while np.linalg.norm(grad) > GRADIENT_ZERO_TOLERANCE:
    h_det = hessian[0,0] * hessian[1,1] - hessian[0,1] * hessian[1,0]

    if h_det  < HESSIAN_ZERO_TOLERANCE:
      hessian = ComputeCorrectedHessian(hessian)
    if iterCount == NEWTON_ITERATION_LIMIT:
      print("Limit")
      break
    
    inverse_hessian = np.linalg.inv(hessian)

    dir = - np.matmul(inverse_hessian, grad)

    theta  += computeStepSize(dir, tau, gamma, sigma ,c21, grad , K, theta)
    
    grad    = ComputeGradient(theta, c21,K)
    hessian = ComputeHessian(theta, c21,K)
    
    iterCount +=1
    print(np.linalg.norm(grad) )
