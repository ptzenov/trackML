import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits import mplot3d


wdir = os.getcwd()

hits_cols = "hit_id,x,y,z,volume_id,layer_id,module_id,event_name"
particle_cols = "particle_id,vx,vy,vz,px,py,pz,q,nhits,event_name"
truth_cols = "hit_id,particle_id,tx,ty,tz,tpx,tpy,tpz,weight,event_name"
cells_cols = "hit_id,ch0,ch1,value,event_name"

hits_df = pd.DataFrame(columns = hits_cols.split(","))
particle_df = pd.DataFrame(columns=particle_cols.split(","))
truth_df =  pd.DataFrame(columns = truth_cols.split(","))
cells_df = pd.DataFrame(columns= cells_cols.split(','))



#  todo - take data for only single event!
#  for some reason events repeat themselves


events2probe = ['event000001000','event000001001','event000001002','event000001003','event000001004']
fulldir = os.getcwd()+"/data/train_sample"
ctr = 1

for evnt in events2probe:
    fname = fulldir+"/"+evnt
    tmp = pd.read_csv(fname+'-hits.csv')
    tmp['event_name'] = evnt
    hits_df = pd.concat([hits_df,tmp],axis = 0)
    tmp = pd.read_csv(fname+'-truth.csv')
    tmp['event_name'] = evnt
    truth_df = pd.concat([truth_df,tmp],axis = 0)

df  = pd.merge(hits_df,truth_df,how = 'left', on = ['hit_id','event_name'])
df = df[df['particle_id']!= 0]
grouped = df.groupby(['event_name','particle_id'])

#  now plot the trajectory of several randomly chosen particles for each event
for event2plot in events2probe:
    # get all particles for that event except particle 0 since it exhibits strange behaviour !
    particles_ = df[(df['event_name'] == event2plot)]['particle_id'].values
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    legend = []
    for i in range(0,15):
        # select a particular group
        pid = np.random.randint(0,len(particles_))
        grp = grouped.get_group((event2plot,particles_[pid]))
        #now plot the group
        grp.sort_values('hit_id', inplace=True )
        ax.plot(grp.tx,grp.ty,grp.tz)
        legend.append('traj_'+str(pid))

    ax.legend(legend)
    fig.savefig(event2plot+'_random_trajectories.png')

#  Numerical calculatoin of the trajectory curvature
#  the curvature at a point t for a reparametrized 3D curve
#  is equal to |d^2r/dt^2|
#  Calculates the curvature of the particle trajectory

def calc_curvature(data_fr):
    data_fr.sort_values('hit_id',inplace=True)
    x = data_fr.tx
    y = data_fr.ty
    z = data_fr.tz
    ddx  = np.diff(np.diff(x))
    ddy  = np.diff(np.diff(y))
    ddz  = np.diff(np.diff(z))
    kappa = np.sqrt(ddx**2 + ddy**2 + ddz**2).sum() # total curvature ?!?
    return kappa


#calculate the curvatures
curvatures = grouped.apply(calc_curvature)
#plot a histogram
event000001000 = curvatures['event000001000']
event000001000.hist(bins = np.histogram(range=[0,event000001000.max]))