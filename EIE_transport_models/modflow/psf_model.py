####################################################################
## psf_model.py 
## Author: Andy Banks 2019 - University of Kansas Dept. of Geology
#####################################################################
# code to simulate groundwater flow during EIE using MODFLOW
# MODFLOW model is constructed using FloPy
# has paralell features for simulating several distinct models at once
#   > good for incorporating multiple realizations of heterogeneity fields
#   > good for experienting with different EIE design parameters (i.e. pumping sequences)
    

################# Begin Code ###########################

# import python packages
import numpy as np
import flopy
import sys
import matplotlib.pyplot as plt


## function to convert starting location data from XYZ to cell# and local xyz
def XYZtoCell(X,Y,grid_ref):
    cells = grid_ref.get_rc(x = X,y= Y)
    npt = len(X)
    row = np.zeros((npt,),dtype=int)
    col = np.zeros((npt,),dtype=int)   
    xloc = np.zeros((npt,),dtype=float)
    yloc = np.zeros((npt,),dtype=float)
    for particle in np.arange(0,npt):
        row[particle,] = cells[0][particle]
        col[particle,] = cells[1][particle] 
        verts = grid_ref.get_vertices(row[particle,],col[particle,])        
        xverts = [verts[0][0],verts[1][0],verts[2][0],verts[3][0]]
        yverts = [verts[0][1],verts[1][1],verts[2][1],verts[3][1]]      
        dx = max(xverts)-min(xverts)
        dy = max(yverts)-min(yverts)
        xcol = max(xverts)
        ycol = max(yverts)
        xloc[particle,] = 1-(xcol - X[particle])/dx
        yloc[particle,] = 1-(ycol - Y[particle])/dy
    return row, col, xloc, yloc


def gen_psf_model(data):

    # INPUT : accapts dict (data) containinf the name of the modflow model
    # --code can be changed to accept other model parameters in data dict 
    modelname = data['modelname']
            
    # Model domain and grid definition
    ztop = 0 		# top elevation 
    zbot = -10 		# bottom elevation 
    nlay = 1 	        # n layers 
    nrow = 1201 		# n rows 
    ncol = 1201              # n cols


    L = 25 # distance in meters between origin and well(s)

    delr = 0.25*np.ones([ncol,])
    delc = 0.25*np.ones([nrow,])

    Lx = np.sum(delr);
    Ly = np.sum(delc);

    delv = (ztop - zbot) / nlay # layer spacing 
    botm = np.linspace(ztop, zbot, nlay + 1) #


    # Variables for the BAS package
    # Note that changes from the previous tutorial!
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    ibound[:,:,0]  =  -1  # constant heads on E  border
    ibound[:,:,-1] =  -1  # constant heads on W  border
    ibound[:,0,:]  =  0  # constant heads on N  border
    ibound[:,-1,:] =  0  # constant heads on S  border

    strt = abs(ztop-zbot) * np.ones((nlay, nrow, ncol), dtype=np.float)
    hnoflo = 0
    
    # Time step parameters
    ncycles  = 1 #number of neupauer 12period pumping cycles
    nper_in_cycle = 12 # number of stress periods per cycle
    nper = nper_in_cycle*ncycles + 1 # add extra SS step in the beginning

    perlen = 6.25*np.ones((nper,),dtype=np.float32) #add extra position for SS stress period
    perlen[0] = 1; # set SS stress period length shorter
    perlen = perlen.tolist()

    nstp = 10*np.ones((nper,),dtype=np.int32)
    nstp[0]= 1; # set SS stress period length shorter
    nstp = nstp.tolist()

    steady = np.logical_and(np.arange(nper,)>-1,np.arange(nper,)<1) # set all but first stress period to transient 
    steady = steady.tolist();

    itmuni = 4 # time units 0 = undefined , 1= seconds, 2 = min , 3 = hours,  4 = days
    lenuni = 2 # length units 2 =  meters 

    xul = -Lx/2 # x coordinate for upper left corner of grid
    yul =  Lx/2 # y coordinate for upper left corner of grid



    # Flopy objects
    modelname = modelname
    mf = flopy.modflow.Modflow(modelname, exe_name='mf2005')
    dis = flopy.modflow.ModflowDis(mf, nlay = nlay, nrow = nrow, ncol = ncol, delr=delr, delc=delc,top=ztop, botm=botm[1:],
                                   nper=nper, perlen=perlen, nstp=nstp,steady=steady,
                                   itmuni = itmuni ,lenuni = lenuni,
                                   xul = xul, yul=yul )

    grid_ref = flopy.utils.reference.SpatialReference(delr=mf.dis.delr, delc=mf.dis.delc, lenuni=mf.dis.lenuni ,xul = xul,yul = yul)
    
    hk = 0.5
    vka = 1.
    sy = 0.1
    ss = (10**-5)/abs(ztop - zbot)
    laytyp = 0 # 0= confined 

    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt,hnoflo = hnoflo)
    lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=vka, sy=sy, ss=ss, laytyp=laytyp,ipakcb=53)
    pcg = flopy.modflow.ModflowPcg(mf)

    # Create the well package

    # determine what cells to place wells in
    wellX = [0, 0, L, -L]
    wellY = [L, -L, 0, 0]
    well_row,well_col,wlocx,wlocy = XYZtoCell(wellX,wellY,grid_ref)

    Qcoef = 1

    # single intitial pumping cycle with 0 pumping rates for SS stress period
    # North 
    pumping_rate1 = Qcoef*np.array([0 , 0   , 0   , 0    , 0    , 0    , 0    , 0    , 875 , 0    , -750 , 0    , -350])
    # South
    pumping_rate2 = Qcoef*np.array([0 , 0   , 0   , 0    , 0    , 0    , 0    , 875  , 0   , -250 , 0    , -400 , 0   ])
    # East 
    pumping_rate3 = Qcoef*np.array([0 , 0   , 875 , 0    , -750 , 0    , -350 , 0    , 0   , 0    , 0    , 0    , 0   ])
    # West 
    pumping_rate4 = Qcoef*np.array([0 , 875 , 0   , -250 , 0    ,- 400 , 0    , 0    , 0   , 0    , 0    , 0    , 0   ])


    # initialize dict for well stress period data 
    sp_data = {}

    # wel data for stress period 0 (SS)
    well_sp0  = [[0, well_row[0] , well_col[0] , pumping_rate1[0]],
                 [0, well_row[1] , well_col[1] , pumping_rate2[0]],
                 [0, well_row[2] , well_col[2] , pumping_rate3[0]],
                 [0, well_row[3] , well_col[3] , pumping_rate4[0]]]
    sp_data = {0:well_sp0}


    # loop over all but initial SS stress period indicies
    per_cnt = 1 # counter for indexing stress periods (start at 1 to avoid overwriting SS stress period  0)
    for cycle in np.arange(0,ncycles):
        for per in np.arange(1,nper_in_cycle+1):
                        #[lay,row,col,Qrate]
            well_sp  =  [[0, well_row[0] , well_col[0] , pumping_rate1[per]],
                         [0, well_row[1] , well_col[1] , pumping_rate2[per]],
                         [0, well_row[2] , well_col[2] , pumping_rate3[per]],
                         [0, well_row[3] , well_col[3] , pumping_rate4[per]]]
        
            sp_data[per_cnt] = well_sp
            per_cnt = per_cnt + 1                 
    wel = flopy.modflow.ModflowWel(mf, stress_period_data=sp_data)
    # Output control
    sp_data_OC = {}
    for kper in range(nper):
        for kstp in range(nstp[kper]):
            sp_data_OC[(kper, kstp)] = ['save head',
                                        'save drawdown',
                                        'save budget']          
    oc = flopy.modflow.ModflowOc(mf, stress_period_data=sp_data_OC,
                                 compact=True)
    # Write the model input files
    mf.write_input()
    # Run the model
    success, mfoutput = mf.run_model(silent=False, pause=False, report=False)
    if not success:
        raise Exception('MODFLOW did not terminate normally.')

    return mf
    





dataA = {'modelname': 'case0'} # data to pass to gen_psf_model
mf = gen_psf_model(dataA) # run model

'''
# Paralell processing options for running multiple models 
if __name__ == '__main__':

    # Define the dataset
    num = [dataA,dataB,dataC,dataD,dataE]

    # Run this with a pool of N agents having a chunksize of M until finished
    agents = len(num)
    chunksize = 1
   
    with Pool(processes=agents) as pool:
         out = pool.map(gen_psf_model, num, chunksize)
         
'''
