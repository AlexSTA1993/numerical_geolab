'''
Created on Nov 5, 2018

@author: Alexandros Stathas

Contains unit tests of ngeoFE applied to Cauchy continua in 1D linear elasticity
Checks:
-Convergence
-Generalised force displacement values
-Analytical vs numerical total stress
-Final temperature and pore fluid pressure values
-Rate of temperature and pore fluid pressure increase
'''
import sys
import os
import unittest
from ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.BVP.CAUCHY_THM_ThermoHydroelasticity_0 import THM3D_FEformulation,THM3D_FEproblem

from dolfin import *

from dolfin.cpp.io import HDF5File

import pickle
import numpy as np
from functools import reduce

from ngeoFE_unittests import ngeo_parameters
from ngeoFE_unittests import plotting_params 

reference_data_path = ngeo_parameters.reference_data_path    

# Check if the environment variable or command-line argument is set to activate plots
activate_plots = False

if 'RUN_TESTS_WITH_PLOTS' in os.environ and os.environ['RUN_TESTS_WITH_PLOTS'].lower() == 'true':
    activate_plots = True
elif len(sys.argv) > 1 and sys.argv[1].lower() == 'with_plots':
    activate_plots = True

class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''
        Run FE analysis example
        '''
        cls.notfirsttime=True
        cls.my_FEformulation=THM3D_FEformulation()
        
        dts=[0.1,0.5,1,5]
        dtmax=500
        cls.myFEproblems=[]
        for dt in dts:
            cls.FEproblem=THM3D_FEproblem(cls.my_FEformulation,dt,dtmax)
            cls.FEproblem.give_me_solver_params(scale_t=1.)
            cls.converged1=cls.FEproblem.run_analysis_procedure(reference_data_path)       
            cls.myFEproblems.append(cls.FEproblem)
        
    def test_execution(self):
        '''
        Tests execution and convergence
        '''
        self.assertTrue(self.converged1, "Convergence failed")
    
    def test_generalized_force_displacement_values(self):
        '''
        Tests calculated nodal forces and displacements to values in ./reference_data/Cauchy3D_THM_Thermoelasticity_0.out
        '''
        self.myFEproblems[0].extract_generalized_force_disp()
        values_time1 = self.myFEproblems[0].array_time
        values_gen_force1 = self.myFEproblems[0].array_gen_force
        values_gen_disp1 = self.myFEproblems[0].array_gen_disp
        
        values1=np.concatenate((values_time1, values_gen_disp1, values_gen_force1), axis=1)
        
        # with open(reference_data_path+"Cauchy3D_THM_Thermoelasticity_0.out", "wb") as fp:   #Pickling
        #     pickle.dump(values1,fp)        
        
        with open(reference_data_path+"Cauchy3D_THM_Thermoelasticity_0.out", "rb") as fp:   #Pickling
            values_ref=pickle.load(fp)        
        values_diff=values_ref[:]-values1
    
        equal=abs(np.linalg.norm(values_diff))<=1.e-10
        self.assertTrue(equal, "Not identical time, displacements, forces: "+str(abs(np.linalg.norm(values_diff))))
        #asserts that data are correct
        #if data are correct then plot diagram
        if equal and activate_plots:
            x1=list(values_time1[:].copy())
            y1=list(values_gen_disp1[:,-1].copy())
            x1.insert(0,0)
            y1.insert(0,0)
            
            filepath=reference_data_path+"Diffusion_Analytical.bin"

            with open(filepath,'rb') as f:
                depth=np.fromfile(f,dtype=np.dtype('int32'),count=1)[0]
                dims =np.fromfile(f,dtype=np.dtype('int32'),count=depth)
                analytical_temperature_values =np.reshape(np.fromfile(f,dtype=np.dtype('float64'),
                count=reduce(lambda x,y:x*y,dims)),dims)
        
            time = analytical_temperature_values[1:,0]
            xy =np.intersect1d(time,values_time1)
            # Find indices of the common elements in the original arrays
            x_ind = np.where(np.in1d(time, xy))[0]
            y_ind = np.where(np.in1d(values_time1, xy))[0]
            
            x1=values_time1[y_ind].copy()
            y1=values_gen_disp1[y_ind,-1].copy()
            x2=analytical_temperature_values[x_ind,0]
            y2=analytical_temperature_values[x_ind,1]

            fig, ax1, ax2, color1, color2 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$T_{num}\; [^o $ C$]$',color1='',y2_txt='$T_{an}\; [^o $ C$]$',color2='', title='',mode='2')
            plotting_params.object_plot(x2, y2, y1, ax1, ax2, mode='2',color1=color1,color2=color2,label_string='')
            plotting_params.show_plot()                       
            fig, ax1, color1 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$u_z$ [mm]',color1='',y2_txt='',color2='', title='',mode='1')
            
            plotting_params.object_plot(x1, y1, "y1", ax1, 'ax2', mode='1',color1=color1,color2='color2',label_string='')
            plotting_params.show_plot()           


            
    def test_thermal_diffusion(self):  
        self.myFEproblems[0].extract_generalized_force_disp()
        values_time1 = self.myFEproblems[0].array_time

        num_temperature_values=self.myFEproblems[0].array_gen_disp[:,-1]
    
        #write data to binary files
        # with open(reference_data_path+"Cauchy3D_THM_Thermoelasticity_diffusion.out", "wb") as fp:   #Pickling
        #     pickle.dump(num_temperature_values, fp)  
    
        filepath=reference_data_path+"Diffusion_Analytical.bin"

        with open(filepath,'rb') as f:
            depth=np.fromfile(f,dtype=np.dtype('int32'),count=1)[0]
            dims =np.fromfile(f,dtype=np.dtype('int32'),count=depth)
            analytical_temperature_values =np.reshape(np.fromfile(f,dtype=np.dtype('float64'),
            count=reduce(lambda x,y:x*y,dims)),dims)
        
        time = analytical_temperature_values[1:,0]
        xy =np.intersect1d(time,values_time1)
        x_ind = np.where(np.in1d(time, xy))[0]
        y_ind = np.where(np.in1d(values_time1, xy))[0]


        values_diff=np.abs(analytical_temperature_values[x_ind,1]-num_temperature_values[y_ind])#/(analytical_temperature_values[1:,1])

        equal=abs(np.linalg.norm(values_diff))<=1.e-4 #The error depends on the discretization of the domain
        self.assertTrue(equal, "Not identical temperature evolution: "+str(abs(np.linalg.norm(values_diff))))    

        if equal and activate_plots:
            
            fig, ax1, color1 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$\|\\frac{\dot{T}_{an}-\dot{T}}{\dot{T}_{an}}\|$',
                                            color1='k',y2_txt='',color2='', title='',mode='1')
            
            for i, myFEproblem in enumerate(self.myFEproblems):
                myFEproblem.extract_generalized_force_disp() 
                num_temperature_values=myFEproblem.array_gen_disp[:,-1]
    
                values_time1 = myFEproblem.array_time  
                
                xy, x_ind, y_ind=np.intersect1d(np.around(analytical_temperature_values[1:,0],decimals=4),np.around(values_time1,decimals=4),return_indices=True)
                  
                values_diff=np.abs(analytical_temperature_values[x_ind,1]-num_temperature_values[y_ind])#/(analytical_temperature_values[1:,1])
            
                plotting_params.object_plot(xy, values_diff, 'y1', ax1, 'ax2','xy', mode='1',color1='',color2='color2',label_string='')

            plotting_params.show_plot()                       

            path_out=ngeo_parameters.reference_data_path
            filename='Cauchy_THM_diffusion_error_abs_t'
            plotting_params.plot_legends(path_out, fig,filename = filename, mode='1')    

if __name__ == '__main__':
    unittest.main()
