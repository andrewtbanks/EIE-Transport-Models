####################################################################
## advection_sim.py 
## Author: Andy Banks 2019 - University of Kansas Dept. of Geology
#####################################################################
# Code used to simulate advective transport during EIE (engineered Injection and extraction)
# Paralell processing features allow fast computation times
# Advective transport is handled using MODFLOW and MODPATH 
# Simulations are structured for computing measures of spreading (rredistribution of initially nearby particles)
# General Process:
# > Initially a group of particles is distributed on a regular NxN grid withing the region (xmin,xmax)=(ymin,ymax) = (6.25 m, 6.25m)
# > Around each particle on this grid, an additional n = 1000 particles are distributed uniformly on a circle of radius r=0.25m
# > MODPATH timeseries simulation is used to track the particles throughout the EIE sequence.
# > Input data is structured so that particle trajectories may be computed in paralell.  
################# Begin Code ###########################

# import python packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import flopy
import flopy.utils.binaryfile as bf
import flopy.utils.reference  as srf
from modpath_functions import genGridPts
from modpath_functions import genCirclePts
from modpath_functions import XYZtoCell
from multiprocessing import Pool # paralell processing package




def tser_sim(mp_data):
    # Handles advective transport  
    # Shell for MODPATH timeseries simulations and output
    # make directory for current modpath model

    ## INPUT -- dict mp_data with following keys
    # mp_modelname : string containing name of modpath model
    # mf_path : path to root directory of modflow model
    # mf_modelname : name of modflow model
    # n_particles: number of particles to release
    # sloc_raw : n_particles by 6 array with [row,col,lay,xloc,yloc,zloc]
    # times : [t0,tf] , list of initial and final time

    mp_modelname = mp_data['mp_modelname']
    mf_path      = mp_data['mf_path'] # path to modflow model working directory
    
    # create new directory for current modpath model
    created   = 0
    mp_mdl_dir = os.getcwd() + os.sep + 'modpath'+os.sep+ 'lls' +os.sep+ mp_modelname
    for files in os.listdir(os.getcwd() + os.sep + 'modpath'+os.sep+ 'lls' ): # names of files in current working directory
        #print(files)
        if files == mp_modelname:            # if exists
             shutil.rmtree(mp_mdl_dir)                   # delete
             os.makedirs(mp_mdl_dir)                     # recreate
             created = 1
    if created == 0:                                    # otherwise 
             os.makedirs(mp_mdl_dir)                     # create



    # create modpath object group  
    mp  =  flopy.modpath.Modpath(modelname    = mp_modelname,
                                 exe_name     = 'mp6', #os.getcwd()+os.sep+'mp6',
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
    simtype             = 3 # 1 = endpoint , 2 = pathline, 3 = timeseries
    TrackingDirection   = 1 # 1 = forward  , 2 = backward
    StopTimeOption      = 3 # 3 = specify stop time
    ReferenceTimeOption = 1 # 1 = time value,  2 = [tstep,per,fraction]
    TimePointOption     = 3 # 3 = An array of time point values is specified.
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


    # write input  and run model
    mp.write_input()
    mp.run_model(silent=True,pause=False, report=False, normal_msg='normal termination')

    # read and save timeseries output
    tserobj    = flopy.utils.TimeseriesFile(mp_mdl_dir+os.sep+mp_modelname+'.mp.tim_ser')
    tser_data = tserobj.get_alldata()
    np.save(mp_mdl_dir+os.sep+mp_data['outfilename'],tser_data) # save pathlines  to .npy file
    tser_data = np.load(mp_mdl_dir+os.sep+mp_data['outfilename']+'.npy')
    x,y = grid_ref.transform(tser_data['x'],tser_data['y'])
    tser_data['x'] = x
    tser_data['y'] = y
    np.save(mp_mdl_dir+os.sep+mp_data['outfilename'],tser_data) # save pathlines  to .npy file 
    
    
    # read and save endpoint output
    epobj = flopy.utils.EndpointFile(mp_mdl_dir+os.sep+mp_modelname+'.mpend')
    ep_data = epobj.get_alldata()
    final_pos = np.zeros([n_particles,3],dtype = float)
    initial_pos = np.zeros([n_particles,3],dtype = float)
    for particle in np.arange(0,n_particles):

        endpts = ep_data[particle].tolist()
        XYZfinal = endpts[26:29]
        XYZinit  = endpts[14:17]
        xf,yf = grid_ref.transform(XYZfinal[0],XYZfinal[1])
        x0,y0 = grid_ref.transform(XYZinit[0],XYZinit[1])
        initial_pos[particle,:] = [x0,y0,XYZinit[2]]
        final_pos[particle,:] = [xf,yf,XYZfinal[2]]
        
    ep_out = [initial_pos,final_pos] # output : [initial positions,final positions, file path to output]
    np.save(mp_mdl_dir+os.sep+mp_data['mp_modelname']+'_epdata',ep_out) # save endpoint data to .npy file 


    # return timeseries output
    return tser_data 


################# Initalize modeling variable ###########

# load base modflow model
mf_path      = os.getcwd()+os.sep+'modflow'
mf_modelname = 'case0'
mf           = flopy.modflow.Modflow.load(mf_path+os.sep+mf_modelname+ '.nam')               # import modflow object
hdobj        = bf.HeadFile(mf_path+os.sep+mf_modelname+'.hds', precision = 'single')         # import heads file as flopy object ( output from modflow)
mf_times_raw     = hdobj.get_times()# get list of valid solution times from modflow heads file

# create reference grid object corresponding to modflow model
xul = -sum(mf.dis.delr)/2                   # modflow model spatial domain upper left x coordinate 
yul =  sum(mf.dis.delc)/2                   # modflow model spatial domain upper left y coordinate
grid_ref = srf.SpatialReference(delr   = mf.dis.delr,   
                                delc   = mf.dis.delc,
                                lenuni = mf.dis.lenuni,
                                xul    = xul,
                                yul    = yul)
mf_times = [mf_times_raw[i] for i in np.arange(0,len(mf_times_raw),10)] # get modflow times corresponding to end of EIE steps 
L = 25 # distance from origin to each pumping well
Nx = 21 # number of equally spaced material volumes in x-dir (cols)
Ny = 21 # number of equally spaced material volumes in y-dir (rows)
xlim = 0.25*L #right and left margins of grid bounding box
ylim = 0.25*L #top and bottom  margins of grid bounding box

# grid of central trajectories for each parcel
[Xg,Yg] = np.meshgrid(np.linspace(-xlim,xlim,Nx),np.linspace(-ylim,ylim,Ny))

Xv = np.reshape(Xg,(Nx*Ny))
Yv = np.reshape(Yg,(Nx*Ny))

# grid of central trajectories for each parcel
[Xg,Yg] = np.meshgrid(np.linspace(-xlim,xlim,Nx),np.linspace(-ylim,ylim,Ny))

Xv = np.reshape(Xg,(Nx*Ny))
Yv = np.reshape(Yg,(Nx*Ny))

# simulate grid of particles
Z = 5*np.ones(np.shape(Xv))
row,col,lay,xloc,yloc,zloc = XYZtoCell(Xv,Yv,Z,mf,grid_ref)
sloc_raw = [row,col,lay,xloc,yloc,zloc]
times = mf_times
rname = 'case0_grid'
mp_data={'mp_modelname': rname,
         'mf_modelname': mf_modelname,
              'mf_path': mf_path,
          'n_particles': Nx*Ny,
             'sloc_raw': sloc_raw,
                'times': times,
         'outfilename' : rname+'_tserdata'}

t = tser_sim(mp_data) # run simulation for grid of particles



# simulate trajectories for paarticles around each grid point using ParPool
r = 0.01*L # radius of each parcel
npt_par = 1000 # number of particles in each parcel
def shell(n):
    # shell for parPool - takes single integer n which identifies the grid point that this group of particles surrounds

    # identify central grid point
    xc = Xv[n]
    yc = Yv[n]
    
    x,y = genCirclePts(xc,yc,r,npt_par) # generate npt = 1000 initial positions around central point


    # initilize input data for MODPATH timeseries simulation (tser_sim)
    x = np.append(x[0],x) 
    y = np.append(y[0],y)
    z = 5*np.ones(np.shape(x))

    npt = len(x)
    row,col,lay,xloc,yloc,zloc = XYZtoCell(x,y,z,mf,grid_ref)
    sloc_raw = [row,col,lay,xloc,yloc,zloc]
    times = mf_times
    rname = 'case0_parcel_'+str(n)

    mp_data={'mp_modelname': rname,
             'mf_modelname': mf_modelname,
                  'mf_path': mf_path,
              'n_particles': npt,
                 'sloc_raw': sloc_raw,
                    'times': times,
             'outfilename' : rname+'_tserdata'}

    t = tser_sim(mp_data)
    


input_nums = np.arange(0,Nx*Ny)  # input for parPool - each task is computing the trajectories for npt=1000 trajectories surrouding every gridpoint    

if __name__ == '__main__':
    agents = 32 #number of processors
    chunksize = 1 # tasks delegated to each processor 
    with Pool(processes=agents) as pool:
         out = pool.map(shell,input_nums, chunksize)
         






















