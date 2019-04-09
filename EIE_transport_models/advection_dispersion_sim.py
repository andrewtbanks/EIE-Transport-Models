####################################################################
## advection_dispersion_sim.py 
## Author: Andy Banks 2019 - University of Kansas Dept. of Geology
#####################################################################
# Code used to simulate advective and dispersive transport during EIE (engineered Injection and extraction)
# Advective transport is handled using MODFLOW and MODPATH
# Dispersive transport is simulated using a random walk process
# General Process:
# - C1 particles represent treatment solution and surrounding contaminated groundwater
# - Treatment solution (C1)is initalized as collection of particles uniformaly spaced throughout a circular region of radius 6.25m, centered at the origin (x,y) = (0,0)
# - Contaminated groundwater(C2)is initalized as collection of particles uniformaly spaced throughout an annular region of outer radius 12.5m and inner radius of 6.25 m, centered on the tretment (C1) particles
# - Prior to each timestep, dispersive transport is modeled by superimposing random displacements in the direction of the local velocity vector, and in the direction perpendicular to the local velocity vector
# - After displacements due to dispersion have been added, MODPATH is used to simulate advective transport up to the next timestep

################# Begin Code ###########################

# import python packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import shutil
import flopy
import flopy.utils.binaryfile as bf
import flopy.utils.reference  as srf
from flopy.utils.postprocessing import get_transmissivities, get_water_table, get_gradients, get_saturated_thickness
from modpath_functions import genGridPts
from modpath_functions import genCirclePts
from modpath_functions import XYZtoCell
import numpy as np
import math

# set seed for reproducibility 
np.random.seed(seed=6)

def makefig():
    # function for making a figure and axis quickly 
    figmake = plt.figure()
    axmake = figmake.gca()
    axmake.set_aspect('equal')
    return figmake, axmake

def gen_grid_circle(xc,yc,N,r):
  # generates a NxN grid of particles shaped in a circle of radius r, cetered about (x0,y0)

  
  xl = xc + r
  yl = yc + r
  N = 101 # choose N = 101 for these simulations
  
  # initial square grid
  Xinit,Yinit = np.meshgrid(np.linspace(-xl,xl,N),np.linspace(-yl,yl,N))

  Xinit = np.reshape(Xinit,N**2)
  Yinit = np.reshape(Yinit,N**2)
  
  # eliminate outer points to form circle (remove any points a distance greater than r from the origin)
  dist_cent_outer_cont = np.sqrt(Xinit**2 + Yinit**2)# compute distance between each point and the center
  cont_outer = np.where(dist_cent_outer_cont<=r)[0]# eliminate points outside outer radius
  XinitC = Xinit[cont_outer]
  YinitC = Yinit[cont_outer]

  # determing points corresponding to the contaminant (c2) - initially the outer annular region of outer radius r and inner radius of r/2 m, 
  dist_cent_inner_cont = np.sqrt(XinitC**2 + YinitC**2)# compute distance between each point and the center
  cont_inner = np.where(dist_cent_inner_cont>0.5*r)[0]# eliminate points inside interface radius (0.25/L)
  Xcont = XinitC[cont_inner]
  Ycont = YinitC[cont_inner]
  
# determing points corresponding to the contaminant (c2) - initially the circular region of radius r 
  treat_inner = np.where(dist_cent_inner_cont<=0.5*r)[0]# eliminate points inside radius 
  Xtreat = XinitC[treat_inner]
  Ytreat = YinitC[treat_inner]

  # order positions so the treatement solution particles are the first entries   
  Xout = np.append(Xtreat,Xcont)
  Yout = np.append(Ytreat,Ycont)

  return Xout,Yout


def ep_sim(mp_data):
    # Handles advective transport  
    # Shell for MODPATH endpoint simulations and output
    
    
    ## INPUT -- dict mp_data with following keys
    # mp_modelname : string containing name of modpath model
    # mf_path : path to root directory of modflow model
    # mf_modelname : name of modflow model
    # n_particles: number of particles to release
    # sloc_raw : n_particles by 6 array with [row,col,lay,xloc,yloc,zloc]
    # times : [t0,tf] , list of initial and final time

    mp_modelname = mp_data['mp_modelname'] # name of modpath model
    mf_path      = mp_data['mf_path'] # path to modflow model working directory
    
    # create new directory for current modpath model
    created   = 0
    mp_mdl_dir = os.getcwd() + os.sep + 'modpath'+os.sep+ mp_modelname
    for files in os.listdir(os.getcwd() + os.sep + 'modpath'): # names of files in current working directory
        #print(files)
        if files == mp_modelname:            # if exists
             shutil.rmtree(mp_mdl_dir)                   # delete
             os.makedirs(mp_mdl_dir)                     # recreate
             created = 1
    if created == 0:                                    # otherwise 
             os.makedirs(mp_mdl_dir)                     # create


    # create modpath object group  
    mp  =  flopy.modpath.Modpath(modelname    = mp_modelname,
                                 exe_name     = 'mp6', 
                                 modflowmodel = mf,
                                 head_file    = mf_path+os.sep+mf_modelname+'.hds',
                                 budget_file  = mf_path+os.sep+mf_modelname+'.cbc',
                                 dis_file     = mf_path+os.sep+mf_modelname+'.dis',
                                 model_ws     = mp_mdl_dir)
    mp.budget_file = mf_path+os.sep+mf_modelname+'.cbc'
    mp.head_file = mf_path+os.sep+mf_modelname+'.hds'

    mpbas = flopy.modpath.ModpathBas(model       = mp,
                                     hnoflo      = mf.bas6.hnoflo,
                                     hdry        = mf.lpf.hdry,
                                     def_face_ct = 0,
                                     bud_label   = None,
                                     def_iface   = None,
                                     laytyp      = mf.lpf.laytyp[0],
                                     ibound      = mf.bas6.ibound[0],
                                     prsity      = 0.25,
                                     prsityCB    = 0.25,
                                     extension   = 'mpbas',
                                     unitnumber  = None)

    # option flags for writing template MPSIM file
    simtype             = 1 # 1 = endpoint , 2 = pathline, 3 = timeseries
    TrackingDirection   = 1 # 1 = forward  , 2 = backward
    StopTimeOption      = 3 # 3 = specify stop time
    ReferenceTimeOption = 1 # 1 = time value,  2 = [tstep,per,fraction]
    TimePointOption     = 1 # must be =1 for endpoint, 3 = An array of time point values is specified.
    option_flags = [simtype, TrackingDirection, 1, 1, ReferenceTimeOption, StopTimeOption, 2, TimePointOption, 1, 1, 1, 1]

    # create template MPSIM and SLOC object 
    sim = flopy.modpath.ModpathSim(model        = mp,
                                   option_flags = option_flags,
                                   stop_time    = 'None',
                                   time_ct      = len(mp_data['times']),
                                   time_pts      = mp_data['times'])
    
    sloc = flopy.modpath.mpsim.StartingLocationsFile(model=mp,verbose = True )
    sloc_raw = mp_data['sloc_raw']  # [row,col,lay,xloc,yloc,zloc] initial position
    n_particles = mp_data['n_particles']
    sloc_data = sloc.get_empty_starting_locations_data(npt=n_particles)
    label_prefix = 'p'
    particle_labels = [label_prefix+'{}'.format(i) for i in range(1, n_particles+1)]
    sloc_data['label']= particle_labels
    for particle in np.arange(0,n_particles):
        sloc_data[particle]['i0']    = sloc_raw[0][particle]
        sloc_data[particle]['j0']    = sloc_raw[1][particle]
        sloc_data[particle]['k0']    = sloc_raw[2][particle]-1
        sloc_data[particle]['xloc0'] = sloc_raw[3][particle]
        sloc_data[particle]['yloc0'] = sloc_raw[4][particle]
        sloc_data[particle]['zloc0'] = sloc_raw[5][particle]

        
    sloc.data = sloc_data
    sim.ref_time = mp_data['times'][0]
    sim.stop_time = abs(mp_data['times'][-1] - mp_data['times'][0])

    # write input and run model 
    mp.write_input()
    
    mp.run_model(silent=True,pause=False, report=True, normal_msg='normal termination')
    
    # read and save endpoint output
    epobj = flopy.utils.EndpointFile(mp_mdl_dir+os.sep+mp_modelname+'.mpend')
    ep_data = epobj.get_alldata()

    ep_data_out = np.zeros(n_particles, dtype=[('x', float, 1),
                                           ('y', float, 1),
                                           ('row', float, 1),
                                           ('col', float, 1),
                                           ('species',str, 2)])
    
    final_pos = np.zeros([n_particles,3],dtype = float)
    initial_pos = np.zeros([n_particles,3],dtype = float)
    for particle in np.arange(0,n_particles):

        endpts = ep_data[particle].tolist()
        XYZfinal = endpts[26:29]
        XYZinit  = endpts[14:17]

        RCfinal  = endpts[18:21]
        
        xf,yf = grid_ref.transform(XYZfinal[0],XYZfinal[1])
        x0,y0 = grid_ref.transform(XYZinit[0],XYZinit[1])
        initial_pos[particle,:] = [x0,y0,XYZinit[2]]
        final_pos[particle,:] = [xf,yf,XYZfinal[2]]

        ep_data_out['x'][particle] = xf
        ep_data_out['y'][particle] = yf
        ep_data_out['row'][particle] = RCfinal[1]
        ep_data_out['col'][particle] = RCfinal[2]
        
        
    #ep_out = [initial_pos,final_pos] # output : [initial positions,final positions, file path to output]
    
    # return endpoint data   
    return  ep_data_out


def dispersion_step(ep_data,tstep):
    ## Handles dispersive transport
    ## superimpose random displacements in the direction of the local velocity vector, and in the direction perpendicular to the local velocity vector
    ## displacements randomly generated from normal distribution with variance of
    # > 2*aL*|v|*dt in the direction of the local velocity vector
    # > 2*aTH*|v|*dt in the direction perpendicular to the local velocity vector
    
    # returns xdisp,ydisp - ew positions of particles after dispersion 
    
    
    
    dt = 6.25; # length of timestep (days)
    aL = 0.05; # longitudinal dispersivity (m)
    aTH = 0.005; #  transverse dispersivity

    npt = len(ep_data) # number of particles
    x_disp = [] # list to dtore the x position for each particle after dispersion 
    y_disp = [] # list to dtore the y position for each particle after dispersion 
    
    for particle in np.arange(0,npt):
        x = ep_data[particle]['x'] # x position of each particle at the current timestep
        y = ep_data[particle]['y']# y position of each particle at the current timestep
        row = int(ep_data[particle]['row']) # row position of each particle at the current timestep
        col = int(ep_data[particle]['col']) # col position of each particle at the current timestep

    # call del h vector corresponding to particle R-C position
        dhdx = hdata[tstep]['dhdx'][row,col]
        dhdy = hdata[tstep]['dhdy'][row,col]
        delh = np.array([dhdx,dhdy],dtype = float)

    # form hydraulic conductivity tensor 
        K = np.array([[Kx[row,col],0],[0,Ky[row,col]]],dtype = float)

    # compute local velocity vector
        porosity = 0.25; # aquifer porosity
        v = -np.dot(K,delh)/porosity

    # compute vector perpendicular to v
        vperp = np.array([-v[1],v[0]])
        
    # compute magnitude of local velocity vector
        vmag = np.linalg.norm(v)

    #  normalize velocity vector to have unit length, d othe same for perpendicular vector 
        vnrm = v/vmag
        vnrm_perp = vperp/vmag



    # compute random displacments
        varL = 2*aL*vmag*dt
        varTH = 2*aTH*vmag*dt
        mean = 0
        randL = np.random.normal(loc = mean, scale = np.sqrt(varL),size = 1)
        randTH = np.random.normal(loc = mean, scale = np.sqrt(varTH),size = 1)
        dispLvec = vnrm*randL # displacement vector in direction of local velocity vector 
        dispTHvec = vnrm_perp*randTH # displacement vector in direction perpendicular to local velocity vector
        dispvec = dispLvec + dispTHvec
        x_disp.append(x+dispvec[0])
        y_disp.append(y+dispvec[1])
        
        # plot stuff
        plot = 0
        if plot == 1:
            xvec = x+vnrm[0]
            yvec = y+vnrm[1]
            ax.scatter(x,y, c = 'black')
            ax.scatter(xvec,yvec, c = 'red')
            ax.plot([x,xvec],[y,yvec], c = 'black')


            xvec = x+vnrm_perp[0]
            yvec = y+vnrm_perp[1]
            ax.scatter(xvec,yvec, c = 'green')
            ax.plot([x,xvec],[y,yvec], c = 'black')


            xvec = x+dispvec[0]
            yvec = y+dispvec[1]
            ax.scatter(xvec,yvec, c = 'blue')
            ax.plot([x,xvec],[y,yvec], c = 'black', ls = '--')

    return x_disp,y_disp
        


########################
# load modflow model
mf_path      =  os.getcwd()+os.sep+'modflow' # 
mf_modelname = 'case0'
mf           = flopy.modflow.Modflow.load(mf_path+os.sep+mf_modelname+ '.nam')               # import modflow object
lpf          = flopy.modflow.ModflowLpf.load(mf_path+os.sep+mf_modelname+ '.lpf', mf)
hdobj        = bf.HeadFile(mf_path+os.sep+mf_modelname+'.hds', precision = 'single')         # import heads file as flopy object ( output from modflow)
cbb          = bf.CellBudgetFile(mf_path+os.sep+mf_modelname+'.cbc')
mf_times_raw     = hdobj.get_times()# get list of valid solution times from modflow heads file
nsteps = 10 # number of solution steps per stress period 
mf_times = [mf_times_raw[i] for i in np.arange(0,len(mf_times_raw),nsteps)]

# create reference grid object corresponding to modflow model
xul = -sum(mf.dis.delr)/2                   # modflow model spatial domain upper left x coordinate 
yul =  sum(mf.dis.delc)/2                   # modflow model spatial domain upper left y coordinate
grid_ref = srf.SpatialReference(delr   = mf.dis.delr,   
                                delc   = mf.dis.delc,
                                lenuni = mf.dis.lenuni,
                                xul    = xul,
                                yul    = yul)
nrow = 1201 
ncol = 1201
L = 25 # distance from origin to each pumping well

################# Computations needed for simulating dispersion 
# form del h
hdata = np.zeros(len(mf_times),dtype = ([ ('heads',float,[nrow,ncol]),
                                          ('time',float,1),
                                          ('dhdx',float,[nrow,ncol]),
                                          ('dhdy',float,[nrow,ncol])]))
porosity = 0.25
dx = mf.dis.delr.array[0]
dy = mf.dis.delc.array[0]
for step in np.arange(0,len(mf_times)):
    
    hdata[step]['time'] = mf_times[step]
    h = hdobj.get_data(totim = mf_times[step],mflay = 0)
    hdata[step]['heads'] = h
    dhdy,dhdx = np.gradient(h,dx,dy)
    dhdy = -dhdy
    
    hdata[step]['dhdy'] = dhdy
    hdata[step]['dhdx'] = dhdx


# form K tensor
Kx = lpf.hk[0].array # hydraulic conductivity along rows
if lpf.chani.array <= 0: #use hani
    
    Ky = lpf.hani[0].array  * Kx  # hydraulic conductivity along columns
else: #use chani
    Ky = lpf.chani.array * Kx
    
if lpf.layvka.array == 0:
    Kz = lpf.vka[0].array  
else:
    Kz = Kx /lpf.vka[0].array


## init advection dispersion sim   
def init():
    # shell for initial advection step
    xc = 0
    yc = 0
    r = 2*0.25*L #outer radius of plume

    ## initial shape circle with circle grid
    N = 101
    ntreatment = 1961
    ncontaminant = 5884
    npt = ntreatment + ncontaminant
    
    x,y = gen_grid_circle(xc,yc,N,r)
    z = 5*np.ones(np.shape(x))
    row,col,lay,xloc,yloc,zloc = XYZtoCell(x,y,z,mf,grid_ref)


    # dispersion step
    ep_data_init = np.zeros(npt, dtype=[('x', float, 1),
                                        ('y', float, 1),
                                        ('row', float, 1),
                                        ('col', float, 1),
                                        ('species',str, 2)])
    
    ep_data_init['x'] = x
    ep_data_init['y'] = y
    ep_data_init['row'] = row
    ep_data_init['col'] = col
    
    tstep = 1
    xd,yd = dispersion_step(ep_data_init,tstep)

    # advection step - pass this data to ep_sim (MODPATH endpoint simulation)
    row,col,lay,xloc,yloc,zloc = XYZtoCell(xd,yd,z,mf,grid_ref)
    sloc_raw = [row,col,lay,xloc,yloc,zloc]
    times = [mf_times[0], mf_times[1]] 

    rname = 'adv_disp_step1'
    mp_data={'mp_modelname': rname,
             'mf_modelname': mf_modelname,
                  'mf_path':  mf_path,
              'n_particles': npt,
                 'sloc_raw': sloc_raw,
                    'times': times,
             'outfilename' : rname+'_epdata'}
    ep_data = ep_sim(mp_data)

    return ep_data


ep_data = init()
path_data = []
path_data.append(ep_data)


for tstep in np.arange(1,len(mf_times)-1):
    # loop for advective-dispersive transport in subsequent steps
    print(tstep)
    # load current positions
    ep_data = path_data[tstep-1]


    ## dispersion step
    xd,yd = dispersion_step(ep_data,tstep+1)
    z = 5*np.ones(np.shape(xd))


    # advection step - pass this data to ep_sim (MODPATH endpoint simulation)
    npt = len(xd)
    row,col,lay,xloc,yloc,zloc = XYZtoCell(xd,yd,z,mf,grid_ref)
    sloc_raw = [row,col,lay,xloc,yloc,zloc]
    times = [mf_times[tstep], mf_times[tstep+1]]
    rname = 'adv_disp_step'+str(tstep+1)
    mp_data={'mp_modelname': rname,
             'mf_modelname': mf_modelname,
                  'mf_path': mf_path,
              'n_particles': npt,
                 'sloc_raw': sloc_raw,
                    'times': times,
             'outfilename' : rname+'_epdata'}

    ep_data = ep_sim(mp_data)
    path_data.append(ep_data)


# save output to file - for use by reaction_sim
np.save('modpath'+os.sep+'adv_disp_data_case0',path_data)

#import reaction_sim.npy


