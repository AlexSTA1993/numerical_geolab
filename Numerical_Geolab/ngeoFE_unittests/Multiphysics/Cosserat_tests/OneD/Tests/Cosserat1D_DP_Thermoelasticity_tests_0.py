'''
Created on Nov 5, 2018

@author: Alexandros Stathas

Contains unit tests of ngeoFE applied to Cosserat continua in 1D Hydroelasticity.
Checks:
-Convergence
-Generalised force displacement values
-Steady state displacement values
-Diffusion time test
'''
import sys
import os

import unittest
from ngeoFE_unittests.Multiphysics.Cosserat_tests.OneD.BVP.Cosserat1D_Drucker_Prager_Thermo_Elasticity import CosseratTHM1DFEformulation,CosseratTHM1DFEproblem

from dolfin import *

# from dolfin.cpp.io import HDF5File

import pickle
import numpy as np

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
        cls.my_FEformulation=CosseratTHM1DFEformulation()
        
        #first slow loading procedurde
        cls.my_FEproblem1=CosseratTHM1DFEproblem(cls.my_FEformulation)
        cls.my_FEproblem1.give_me_solver_params(scale_t=1.)
        cls.converged1=cls.my_FEproblem1.run_analysis_procedure(reference_data_path) 
        # if cls.converged==True: cls.my_FEproblem.plot_me()
        
    def test_execution(self):
        '''
        Tests execution and convergence
        '''
        self.assertTrue(self.converged1, "Convergence failed")

    def test_generalized_force_displacement_values(self):
        '''
        Tests calculated nodal forces and displacements to values in ./reference_data/Cosserat_1D_DP_force_disp_values.out
        '''
        self.my_FEproblem1.extract_generalized_force_disp()
        values_time1 = self.my_FEproblem1.array_time
        values_gen_force1 = self.my_FEproblem1.array_gen_force
        values_gen_disp1 = self.my_FEproblem1.array_gen_disp

        values1=np.concatenate((values_time1, values_gen_disp1, values_gen_force1), axis=1)

        #with open(reference_data_path+"Cosserat1D_DP_THM_Thermoelasticity.out", "wb") as fp:   #Pickling
        #    pickle.dump(values1,fp)        

        with open(reference_data_path+"Cosserat1D_DP_THM_Thermoelasticity.out", "rb") as fp:   #Pickling
            values_ref=pickle.load(fp)        
        values_diff=values_ref-values1

        equal=abs(np.linalg.norm(values_diff))<=1.e-8 #docker precision is reduced it however better than the accuracy of the equlibrium solver 10e-6 and equal to material solver 10e-8
        self.assertTrue(equal, "Not identical time, displacements, forces: "+str(abs(np.linalg.norm(values_diff))))
        #asserts that data are correct
        #if data are correct then plot diagram
        
        if equal:
            step=int(5./self.my_FEproblem1.slv.dtmax)
            x1=list(values_time1[1:].copy())
            y1=list(values_gen_disp1[:-1:step,-1].copy()[:-1])
            # x1.insert(0,0)
            # y1.insert(0,self.my_FEproblem1.Pressure_loading)
        
            filepath=reference_data_path+'thermal_diffusion_analytical_results.txt'       
            analytical_pressure_values=np.loadtxt(filepath)
            
            f_index=int(self.my_FEproblem1.slv.tmax/5.)+1
            x2=analytical_pressure_values[:f_index,0]
            y2=analytical_pressure_values[:f_index,1]
                    
            fig, ax1, ax2, color1, color2 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$T_{an}$ [$^o$C]',color1='',y2_txt='$T_{num}$ [$^o$C]',color2='', title='',mode='2')
        
            plotting_params.object_plot(x2, y2, y1, ax1, ax2, mode='2',color1=color1,color2=color2,label_string='')
            plotting_params.show_plot()           
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cosserat_1D_Thermoelasticity_Tcalc_Tanal',mode='1')
        
            x1=list(values_time1[:].copy())
            y1=list(values_gen_disp1[:,0].copy())
            x1.insert(0,0)
            y1.insert(0,0)
        
            fig, ax1, color1 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$u_z$ [mm]',color1='',y2_txt='',color2='', title='',mode='1')
        
            plotting_params.object_plot(x1, y1, 'y2', ax1, 'ax2', mode='1',color1=color1,color2='',label_string='')
            plotting_params.show_plot()           
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cosserat_1D_Thermoelasticity_u_anal',mode='1')

if __name__ == '__main__':
    unittest.main()
