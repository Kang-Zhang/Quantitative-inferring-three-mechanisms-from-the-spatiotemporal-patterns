#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16
@author: 
"""
from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
import os, time
from scipy import misc
import imageio
# %% Parameter definition 

# Defining the device that is used
DeviceNr = 0   # 0 = CPU; 1 = Intel; 2 = AMD
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0' # Set to 1 for compiler comments
# parameters
D        =  0.5    # m2/h  The diffusion constant describing the movement of mussels
# The speeding constant Phi 
Phi      =  100.0    # Speeding constant, accelerates mussel growth
# Simulation settings 
length   = 20000.0       # Length of the physical landscape
Size     = 20000        # Size of the 2D grid
BlockSize= 64         # Size of the workgroup with the GPU 

EndTime  = 100.0 # Total simulation time
NumPlots = 50000         # Number of times the figure is updated
dT       = 0.025     # Time step

# Precalculations
dx = length/Size      # Spatial step size
dy = length/Size      # Spatial step size

Grid_Width = Size
Grid_Height = Size

# %% Defining the initial values

M = np.random.rand(Size*Size)*2.0-0.5
dMdt = np.random.rand(Size*Size)
M_host = M.astype(np.float32)
dMdt_host = dMdt.astype(np.float32)

# Start the timer:
start_time = time.time()

# %% Reporting in the simulation on the console

print("");
print(" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ");
print(" * Quantitative inferring three mechanisms               * ");
print(" * from the spatiotemporal patterns                      * ");
print(" * Cahn-Hilliard model    2019                           * ");
print(" * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n");

print(" Current grid dimensions: %d x %d cells\n" % (Grid_Width, Grid_Height));

# %% Setting up the OpenCL context

platform = cl.get_platforms()
Devices  = platform[0].get_devices()
context  = cl.Context([Devices[DeviceNr]])
queue    = cl.CommandQueue(context)

print(" Compute Device: %s\n" % Devices[DeviceNr].name)

mf = cl.mem_flags # Memory flags are set

# The array is allocated on the GPU and the initial values are copied onto it
M_g = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=M_host)
dMdt_g = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=dMdt_host)
# Set up simulation parameters
global_size=M_host.shape
local_size=(BlockSize,)

# %% Defining the OpenCL simulation kernel

# Parameters
Params = """  
    #define D           %1.5f
    #define Phi         %1.5f
    #define dX          %1.5f
    #define dY          %1.5f    
    #define dT          %1.8f
    #define Grid_Width  %d
    #define Grid_Height %d\n\n
    """ % (D,Phi,dx,dy,dT,Grid_Width,Grid_Height)

ComputeCode = """

////////////////////////////////////////////////////////////////////////////////
// Laplacation operator definition, to calculate diffusive fluxes
////////////////////////////////////////////////////////////////////////////////

static float d2_dxy2(__global float* pop)
{
    // Getting thread coordinates on the grid
    size_t current = get_global_id(0);
    size_t row	 = (size_t)(current/Grid_Width);
    size_t column  = current%Grid_Width;
    
    // Computing positions of the neighbors
    size_t left    = row * Grid_Width + column-1;
    size_t right   = row * Grid_Width + column+1;
    size_t top     = (row-1) * Grid_Width + column;
    size_t bottom  = (row+1) * Grid_Width + column;
    
    float retval = ( (pop[left] - 2.0*pop[current] + pop[right]) )
                     /(float)dX/(float)dX +
                   ( (pop[top]  - 2.0*pop[current] + pop[bottom]))
                     /(float)dY/(float)dY;
    
    return retval;
}

///////////////////////////////////////////////////////////////////////////////
// Gradient operator definition, to calculate advective fluxes
///////////////////////////////////////////////////////////////////////////////

static float d_dy(__global float* pop)
{   
	// Getting thread coordinates on the grid
    size_t current = get_global_id(0);
    size_t row	 = (size_t)(current/Grid_Width);
    size_t column  = current%Grid_Width;
    
    size_t top=(row-1) * Grid_Width + column;        
    return ( pop[current] - pop[top] )/(float)dY ;
}

static float d_dx(__global float* pop)
{
	// Getting thread coordinates on the grid
    size_t current = get_global_id(0);
    size_t row	 = (size_t)(current/Grid_Width);
    size_t column  = current%Grid_Width;

	size_t left    = row * Grid_Width + column-1;	
	return (( pop[current] - pop[left] )/(float)dX );
}

///////////////////////////////////////////////////////////////////////////////
// Simulation kernel
///////////////////////////////////////////////////////////////////////////////

__kernel void SimulationKernel (__global float* dMdt, __global float* M)
{
	
    size_t current  = get_global_id(0);
    size_t row      = floor((float)current/(float)Grid_Width);
    size_t column   = current%Grid_Width;
	
    if (row > 0 && row < Grid_Height-1 && column > 0 && column < Grid_Width-1)
    {
    
    dMdt[current] = M[current]*(M[current] - 0.5)*(M[current] - 1.0) - D*d2_dxy2(M);
    
    }
    else
    {
    dMdt[current] = 0;
    }
            
} // End SimulationKernel


__kernel void SimulationKernel2 (__global float* dMdt, __global float* M)
{

size_t current  = get_global_id(0);
size_t row      = floor((float)current/(float)Grid_Width);
size_t column   = current%Grid_Width;

if (row > 1 && row < Grid_Height-2 && column > 1 && column < Grid_Width-2)
{

float dM = d2_dxy2(dMdt);
M[current] = M[current] + (dM)*dT ;

}

// HANDLE Boundaries
else if(row<=1)
//do copy of first row = second last row
{
M[current]=M[(row + Grid_Height - 4) * Grid_Width + column];
}
else if(row>=Grid_Height-2)
//do copy of last row = second row
{
M[current]=M[(row - Grid_Height + 4) * Grid_Width + column];
}
else if(column<=1)
{
M[current]=M[row * Grid_Width + column + Grid_Width - 4];
}
else if(column>=Grid_Width-2)
{
M[current]=M[row * Grid_Width + column - Grid_Width + 4];
}

} // End SimulationKernel2
"""

# Here the kernel is compiled
program = cl.Program(context, Params + ComputeCode).build()

# %% The Simulation loop

# Setting up time profiling
CompuTime = GraphTime = 0

# Starting the loop
for ii in range(NumPlots):
    
    TimePointC = time.time()
    
    # The simulation in executed here for EndTime/NumFrames times
    for jj in range(int(EndTime/dT)):
        
        program.SimulationKernel(queue, global_size, local_size, dMdt_g, M_g)
        program.SimulationKernel2(queue, global_size, local_size, dMdt_g, M_g)
    
    TimePointG = time.time()
    CompuTime += (TimePointG - TimePointC)

    # Get the data from the GPU
    cl.enqueue_copy(queue, M_host, M_g)
    #cl.enqueue_copy(queue, dMdt_host, dMdt_g)
    
    print(ii)
    # Updating the graphs
    #misc.imsave('CH20000%08d.png'%(ii),M_host.reshape(Size,Size))
    imageio.imwrite('CH20000%08d.png'%(ii),M_host.reshape(Size,Size))


    
    GraphTime += (time.time() - TimePointG)

# %% Wrapping up the simulation





# Determining the time that we used for the simulation
elapsed_time = time.time() - start_time    
print(" Simulation took      : %1.1f (s)" % (elapsed_time))
print("   Computation time   : %1.1f (s)" % CompuTime)
print("   Visualization time : %1.1f (s)\n" % GraphTime)

print(" Finished!\n")
