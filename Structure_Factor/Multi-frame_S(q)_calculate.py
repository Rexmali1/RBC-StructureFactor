import numpy as np
from matplotlib import pyplot as plt

def get_gofr(hist,bin_edges,num_particles, box_size):
    rho = num_particles/(box_size**3)
    bin_centers = (bin_edges[1:]+bin_edges[:-1])/2.0
    dr = bin_edges[1]-bin_edges[0]
    denominator = 4.*np.pi*(bin_centers**2)*dr*rho*( num_particles )
    gofr = hist/denominator
    
    return gofr, bin_centers

def Str_Fac(q,g,N,bin_edges):
  dr  = bin_edges[1]-bin_edges[0]
  h   = g[:,0]*(g[:,1]-1)*np.sin(q*g[:,0])*dr
  S   = 1 + ((4*np.pi)/(q))*sum(h)
  return S

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
    
def histogram_distances(distance_list,r, max_dist, bin_size):
    # this is the list of bins in which to calculate
    bins = np.arange(2*r, max_dist+bin_size, bin_size)
    hist, bin_edges = np.histogram( distance_list, bins=bins )
    return hist, bin_edges


#choose a bin size so that this becomes relatively smooth without throwing away too much information
#set max dist to the box size of this configuration *sqrt(2)/2.

test_bin_size = 0.5
max_dist = L/2.0


dist_hist = []
bin_edges = []

for i in range(len(p_s)):
  distance_list_1_pbc = compute_distances_minimum_image(p_s[i,:,:], L)
  dist_hist_1, bin_edges_1 = histogram_distances(distance_list_1_pbc,r, max_dist, test_bin_size)
    
  dist_hist.append(dist_hist_1)
  bin_edges.append(bin_edges_1)
    
dist_hist_mea = np.mean(dist_hist,axis=0)
dist_hist_sts = np.std(dist_hist,axis=0)
bin_edges_mea = np.mean(bin_edges,axis=0)
bin_edges_sts = np.std(bin_edges,axis=0)

gofr, bin_centers = get_gofr( dist_hist_mea, bin_edges_mea, N, L )

g=np.transpose(np.array([bin_centers,gofr]))
q = np.linspace(1, 40.0, num=100)
Sq =[]
for i in range(len(q)):
    Sq_v=Str_Fac(q[i]*r,g,N,bin_edges_mea)
    Sq.append(Sq_v)

plt.plot(q*r,Sq)
plt.ylabel("$S(q \; R)$")
plt.xlabel("$q \; R$")
plt.title("Structure factor with p = "+str(N/(L**3)))


plt.savefig('Structure_factor_with_p='+str(N/(L**3))+'_V='+str(L**3)+'.png')
