import numpy as np
from matplotlib import pyplot as plt

def compute_distances_minimum_image( configuration, L ):
    distance_list = []
    num_particles = configuration.shape[0]
    for i in range(num_particles):
        for j in range(num_particles):
            if i == j: continue
            #dr is a vector (dx,dy,dz)
            dr = configuration[j] - configuration[i]
            #minimum image dr 
            dr = dr - L*np.floor(dr/L+0.5)
            
            #dr2 is a vector (dx*dx,dy*dy,dz*dz)
            dr2 = dr*dr 
            #dist = sqrt( dx^2 + dy^2 + dz^2)
            dist = np.sqrt( dr2.sum() )            
            distance_list.append(dist)
            
    return np.array(distance_list)

distance_list_1_pbc = compute_distances_minimum_image(P, L)

def histogram_distances(distance_list,r, max_dist, bin_size):
    # this is the list of bins in which to calculate
    bins = np.arange(2*r, max_dist+bin_size, bin_size)
    hist, bin_edges = np.histogram( distance_list, bins=bins )
    return hist, bin_edges

test_bin_size = 0.2
max_dist = L/2.0
dist_hist_1_pbc, bin_edges_1_pbc = histogram_distances(distance_list_1_pbc,r,max_dist=max_dist, bin_size=test_bin_size)

def get_gofr(hist,bin_edges,num_particles, box_size):
    rho = num_particles/(box_size**3)
    bin_centers = (bin_edges[1:]+bin_edges[:-1])/2.0
    dr = bin_edges[1]-bin_edges[0]
    denominator = 4.*np.pi*(bin_centers**2)*dr*rho*( num_particles )
    gofr = hist/denominator
    
    return gofr, bin_centers

gofr, bin_centers = get_gofr( dist_hist_1_pbc, bin_edges_1_pbc, N, L )

def Str_Fac(q,g,N,bin_edges):
  dr  = bin_edges[1]-bin_edges[0]
  h   = g[:,0]*(g[:,1]-1)*np.sin(q*g[:,0])*dr
  S   = 1 + ((4*np.pi)/(q))*sum(h)
  return S

g=np.transpose(np.array([bin_centers,gofr]))
q = np.linspace(0.8, 40.0, num=80)
Sq =[]
for i in range(len(q)):
    Sq_v=Str_Fac(q[i]*r,g,N,bin_edges_1_pbc)
    Sq.append(Sq_v)

plt.plot(q*r,Sq)
plt.ylabel("$S(q \; R)$")
plt.xlabel("$q \; R$")
