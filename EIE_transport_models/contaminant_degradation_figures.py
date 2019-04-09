####################################################################
## contaminant_degradation_figures.py 
## Author: Andy Banks 2019 - University of Kansas Dept. of Geology
#####################################################################
# plotting functions for the EIE transport model ##

# import python packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as PathEffects

#function to compute cumulative mass degraded from data produced by reaction_sim.py
def extract_mass_degraded(rxn_case):
    # use results from reaction_sim.py to compute the cumulative percent mass of contaminant (C2) degreaded during EIE
    
    degraded = np.zeros(len(rxn_case), dtype=[('mass_c1',float,1),# mass C1 at each step
                                               ('mass_c2',float,1),# mass C2 at each step
                                               ('mass_c3',float,1),# mass C3 at each step
                                               ('tot_mass',float,1),# total mass at each step (for mass balance check)
                                               ('fdeg',float,1), # cumulative percent mass C2 degreaded
                                               ('fdeg_diff',float,1), # C2 degreadation at each step
                                               ('fext', float, 1)]) # mass extracted by extracting well at each step

    init_c2_mass = float(ncontaminant) # initial mass of contaminant 
    init_c1_mass = float(ntreatment)*4# initial mass of treatment
    
    # mass of each species for each step 
    c1_mass = [] # treatment
    c2_mass = [] # contaminant
    c3_mass = [] # product

    tot_mass = [] # check to make sure total mass isn't changing
    fdeg_mass = [] # cumulative percent contaminant degraded
    fext_mass = [] # cumulative amount of extracted mass (either c1, c2 or c3)
    for tstep in np.arange(0,ntsteps_rxn):

        species = rxn_case[tstep]['species']
        mass = rxn_case[tstep]['mass']

        tot_c1_mass = 0
        tot_c2_mass = 0
        tot_c3_mass = 0
        extracted_mass = 0
        for particle in rxn_case[tstep]:
            if particle['species'] == 'c1':
                tot_c1_mass += particle['mass']
            if particle['species'] == 'c2':
                tot_c2_mass += particle['mass']
            if particle['species'] == 'c3':
                tot_c3_mass += particle['mass']
            if particle['species'] == 'ex':
                extracted_mass += particle['mass']

        c1_mass.append(tot_c1_mass)
        c2_mass.append(tot_c2_mass)
        c3_mass.append(tot_c3_mass)
        tot_mass.append(tot_c1_mass + tot_c2_mass + tot_c3_mass+extracted_mass)
        fdeg_mass.append(100*(init_c2_mass-tot_c2_mass)/init_c2_mass)
        fext_mass.append(extracted_mass)
     
    degraded['mass_c1'] = c1_mass
    degraded['mass_c2'] = c2_mass
    degraded['mass_c3'] = c3_mass
    degraded['tot_mass'] = tot_mass
    degraded['fdeg'] = fdeg_mass
    degraded['fext'] = fext_mass

    fdeg_diff = np.diff(np.append(0,fdeg_mass))
    degraded['fdeg_diff'] = abs(fdeg_diff)

    return degraded

############################
# make a plot showing the distribution of C2, C2 and C3 particles throughtout space after each step of the EIE sequence
def pane_plot_rxn(rxn_case,case_name):
    nrow = 3
    ncol = 4
    fig,ax = plt.subplots(nrows = nrow,
                          ncols = ncol,
                          sharex = 'all',
                          sharey = 'all',
                          figsize = (4,3))
    plt.subplots_adjust(hspace = 0.1, wspace=0.1)
    tstep = 0

    # define well data
    wellX = [0,0,1,-1]
    wellY = [1,-1,0,0]

    if case_name == 'A':
        well_seq = [3,2,3,2,3,2,1,0,1,0,1,0]
        well_rate = [875,875,-250,-750,-400,-350,875,875,-250,-750,-400,-350]
        well_dir  = [0,0,1,1,1,1,0,0,1,1,1,1]

    #print(sum(well_rate))  
    for row in np.arange(0,nrow):
        for col in np.arange(0,ncol):

            # get xy positions for each species
            c1_ind = np.where(rxn_case[tstep]['species']=='c1')[0]
            c1_pos = np.array([rxn_case[tstep]['x'][c1_ind],rxn_case[tstep]['y'][c1_ind]])
            c1_colors = len(c1_ind)*['yellow']

            c2_ind = np.where(rxn_case[tstep]['species']=='c2')[0]
            c2_pos = np.array([rxn_case[tstep]['x'][c2_ind],rxn_case[tstep]['y'][c2_ind]])
            c2_colors = len(c2_ind)*['blue']
            
            c3_ind = np.where(rxn_case[tstep]['species']=='c3')[0]
            c3_pos = np.array([rxn_case[tstep]['x'][c3_ind],rxn_case[tstep]['y'][c3_ind]])
            c3_colors = len(c3_ind)*['green']

            # turn on grid
            ax[row,col].set_axisbelow(True)
            ax[row,col].grid(True, color = 'gray',which='major', axis='both', alpha = 0.4,linestyle = '--', zorder = -1)
            p_size = 0.0005 # particle plot size
            alpha =1 # alpha for particle plot

            
            c2_particles = ax[row,col].scatter(c2_pos[0,:]/L,
                                               c2_pos[1,:]/L,
                                               c = c2_colors, s = p_size, alpha =alpha)
            c1_particles = ax[row,col].scatter(c1_pos[0,:]/L,
                                               c1_pos[1,:]/L,
                                               c = c1_colors, s = p_size, alpha =alpha)

            c1_particles = ax[row,col].scatter(c3_pos[0,:]/L,
                                               c3_pos[1,:]/L,
                                               c = c3_colors, s = p_size, alpha =alpha)
            

            # plot wells
            wells = ax[row,col].scatter(wellX,wellY,
                                        marker = 'o',
                                        s = 10,
                                        edgecolor = 'black',
                                        facecolor = 'white',
                                        linewidth = 0.8, alpha = 0.6)
            # label well rate
            
            wtext = str(well_rate[tstep])+' $m^{3}/d$'    
            wrate = ax[row,col].annotate(wtext,
                                         xy = (0.2,0.005),
                                         xycoords = 'axes fraction',
                                         fontsize = 6,
                                         fontweight = 'heavy')

            #label timestep
            stext = str(tstep+1)
            step = ax[row,col].annotate(stext,
                                        xy = (0.1,0.825),
                                        xycoords = 'axes fraction',
                                        fontsize = 6,
                                        fontweight = 'heavy')

            # label north well on first timestep

            if tstep == 0:
                North_well_label = ax[row,col].annotate('N',
                                                        xy = (0,1.15),
                                                        xycoords = 'data',
                                                        fontsize = 5,
                                                        fontweight = 'normal',
                                                        horizontalalignment = 'center')
            
            # label active well with pumping direction ( down arrow = injection, positive Q )
            if well_dir[tstep] == 0:
                xy0 = (wellX[well_seq[tstep]], 0.1+wellY[well_seq[tstep]])
                xy1 = (wellX[well_seq[tstep]],0.4+wellY[well_seq[tstep]])
            else:
                xy0 = (wellX[well_seq[tstep]],0.4+wellY[well_seq[tstep]])
                xy1 = (wellX[well_seq[tstep]],0.1+ wellY[well_seq[tstep]])

            #print(xy1)
            
            warrow = ax[row,col].annotate('', xy=xy0,
                                 xytext=xy1,
                                 xycoords = 'data',
                                 arrowprops=dict(facecolor='black',headwidth = 3, headlength = 2, width = 0.1))

            
            ax[row,col].tick_params(axis='both', width = 1E-6, pad = 0.5)
            dim = 1.55
            ax[row,col].set_xlim(-dim ,dim )
            ax[row,col].set_ylim(-dim ,dim )
            ticks = [-1,0,1]
            ax[row,col].set_xticks(ticks)
            ax[row,col].set_yticks(ticks)
            ax[row,col].set_xticklabels(ticks, fontsize = 7, fontweight = 'bold')
            ax[row,col].set_yticklabels(ticks, fontsize = 7, fontweight = 'bold')

            ax[row,col].set_aspect('equal')
            if col == 0:
                ax[row,col].set_ylabel('y/L', fontsize = 8, fontweight = 'bold')
            if row == 2:
                ax[row,col].set_xlabel('x/L', fontsize = 8, fontweight = 'bold')
                                
            tstep = tstep+1

#############################
# function to plot the cumulative mass contaminant degraded vs time
def c2_deg_vs_time():
    
    fig = plt.figure(figsize = (5,3))
    ax = fig.gca()

    # plotting stuff
    lw = 1
    lcolor = ['navy','brown','darkolivegreen','teal','coral','olive']
    mecolor = lcolor
    mfcolor = 6*['white']
    marker = 2*['o','^','*']# 5*[None]
    ls = ['-','-','-','--','--','--']
    alpha =6*[0.75]
    markersize = 2*[5,6,8]
    #mfcolor
    label = ['Case A', 'Case B', 'Case C','Case A2', 'Case B2', 'Case C2']
    zorder = [5,4,3,2,1,0]
    tvals = np.arange(1,13)

    ind = 0
    ax.plot(tvals,degA['fdeg'],label = label[ind],lw = lw,linestyle = ls[ind], color = lcolor[ind],marker = marker[ind],markersize = markersize[ind],markeredgecolor = mecolor[ind],markerfacecolor = mfcolor[ind],alpha = alpha[ind], zorder = zorder[ind])
##    ind = 1
##    ax.plot(tvals,degB['fdeg'],label = label[ind],lw = lw, linestyle = ls[ind],color = lcolor[ind],marker = marker[ind],markersize = markersize[ind],markeredgecolor = mecolor[ind],markerfacecolor = mfcolor[ind],alpha = alpha[ind], zorder = zorder[ind])
##    ind = 2
##    ax.plot(tvals,degC['fdeg'],label = label[ind],lw = lw,linestyle = ls[ind], color = lcolor[ind],marker = marker[ind],markersize = markersize[ind],markeredgecolor = mecolor[ind],markerfacecolor = mfcolor[ind],alpha = alpha[ind], zorder = zorder[ind])
##    ind = 3
##    ax.plot(tvals,degA2['fdeg'],label = label[ind],lw = lw, linestyle = ls[ind],color = lcolor[ind],marker = marker[ind],markersize = markersize[ind],markeredgecolor = mecolor[ind],markerfacecolor = mfcolor[ind],alpha = alpha[ind], zorder = zorder[ind])
##    ind = 4
##    ax.plot(tvals,degB2['fdeg'],label = label[ind],lw = lw, linestyle = ls[ind],color = lcolor[ind],marker = marker[ind],markersize = markersize[ind],markeredgecolor = mecolor[ind],markerfacecolor = mfcolor[ind],alpha = alpha[ind], zorder = zorder[ind])
##    ind = 5
##    ax.plot(tvals,degC2['fdeg'],label = label[ind],lw = lw, linestyle = ls[ind],color = lcolor[ind],marker = marker[ind],markersize = markersize[ind],markeredgecolor = mecolor[ind],markerfacecolor = mfcolor[ind],alpha = alpha[ind], zorder = zorder[ind])


    #### format axis

    # change border width 
    for axis in ['top','bottom','left','right']: 
        ax.spines[axis].set_linewidth(2)
        
    # add grid below    
    ax.set_axisbelow(True)
    ax.grid(True, color = 'gray',which='major', axis='both', alpha = 0.4,linestyle = '--', zorder = -1)
    
    # shift axis position
    bscl = 0.6
    pshift = 0.13
    bbox = [0.125+pshift, 0.11 + pshift, bscl*0.9, bscl*0.88] #[xl,yl,w,h]
    ax.set_position(bbox)

    # set axis limits
    ax.set_xlim(0.75,12.25)
    ax.set_ylim(0,90)
    
    # set ticks and ticklabels 
    yticks = np.arange(10,81,10)
    ax.set_yticks(yticks)
    ax.set_xticks(tvals)
    ax.set_xticklabels(tvals,fontweight = 'bold',fontsize = 9)
    ax.set_yticklabels(yticks,fontweight = 'bold',fontsize = 9)
    ax.tick_params(axis='both', width = 1E-6, pad = 0.1)
    
    # set axis labels 
    ax.set_xlabel('Steps of Remediation', fontweight = 'normal', fontsize = 10)
    ax.set_ylabel('Cumulative Percent \n Contaminant Mass Reacted', fontweight = 'normal', fontsize = 8)
    #ax.set_ylabel('Cumulative' + r'$f_{degradation}', fontweight = 'bold', fontsize = 10)
    # add legend 
    handles,labels = ax.get_legend_handles_labels()
    #fig.legend(handles, labels, loc=(0.275,0.6), ncol = 2,fontsize = 6, framealpha = 0.7)

####################################### 
### init general use variables
L = 25 # distance between each pumping well and origin
ntsteps_rxn = 12# number of timesteps
ncontaminant = 5884# number of contaminant solution particles (species 2)
ntreatment = 1961# number of treatement solution particles (species C1)
init_colors = ntreatment*['yellow']+ncontaminant*['blue'] # initial plotting colors assigned to particles
npt_rxn = ncontaminant + ntreatment # total number of particles used in the simulation 


# load reaction and stretching data. also extract mass degraded for each EIE model
mdl_name = 'case0' # model name (assigned in advection_dispersion_sim.py)
fext = '.npy'
fname = 'rxn_data_'+mdl_name+fext
fpath = os.getcwd() +  os.sep +'modpath'+os.sep
rxn_caseA = np.load(fpath+fname)
degA = extract_mass_degraded(rxn_caseA)# compute contaminant degradation 



# generate figures
c2_deg_vs_time()
pane_plot_rxn(rxn_caseA,'A')
plt.show()
