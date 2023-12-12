'''
Created on Nov 5, 2018

@author: Alexandros Stathas

Contains unit tests of ngeoFE applied to Cauchy continua in 3D Hydroelasticity.
Checks:
-Convergence
-Generalised force displacement values
-Steady state displacement values
-Diffusion time test
'''
import sys
import os

import unittest
from ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.BVP.CAUCHY_DP_HM_Hydroelasticity_0 import THM3D_FEformulation,THM3D_FEproblem

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
        cls.my_FEformulation=THM3D_FEformulation()
        
        #first slow loading procedure
        cls.my_FEproblem1=THM3D_FEproblem(cls.my_FEformulation)
        cls.my_FEproblem1.give_me_solver_params(scale_t=1.)
        cls.converged1=cls.my_FEproblem1.run_analysis_procedure(reference_data_path) 
        
    def test_execution(self):
        '''
        Tests execution and convergence
        '''
        self.assertTrue(self.converged1, "Convergence failed")

    def test_generalized_force_displacement_values(self):
        '''
        Tests calculated nodal forces and displacements to values in ./reference_data/Cauchy3D_DP_force_disp_values.out
        '''
        self.my_FEproblem1.extract_generalized_force_disp()
        values_time1 = self.my_FEproblem1.array_time
        values_gen_force1 = self.my_FEproblem1.array_gen_force
        values_gen_disp1 = self.my_FEproblem1.array_gen_disp

        values1=np.concatenate((values_time1, values_gen_disp1, values_gen_force1), axis=1)

        # with open(reference_data_path+"Cauchy3D_DP_THM_Hydroelasticity.out", "wb") as fp:   #Pickling
        #     pickle.dump(values1,fp)        

        with open(reference_data_path+"Cauchy3D_DP_THM_Hydroelasticity.out", "rb") as fp:   #Pickling
            values_ref=pickle.load(fp)        

        values_diff=values_ref-values1

        equal=abs(np.linalg.norm(values_diff))<=1.e-9 #precision in docker is reduced accuracy is smaller than the tolerance of the equilibrium solver and the material solver
        self.assertTrue(equal, "Not identical time, displacements, forces: "+str(abs(np.linalg.norm(values_diff))))
        #asserts that data are correct
        #if data are correct then plot diagram
        
        if equal and activate_plots:
            x1=list(values_time1[9:].copy())
            y1=list(values_gen_disp1[10:,-5].copy())
            x1.insert(0,0)
            y1.insert(0,self.my_FEproblem1.Pressure_loading)
        
            filepath=reference_data_path+'thermal_diffusion_analytical_results.txt'       
            analytical_pressure_values=np.loadtxt(filepath)
        
            x2=analytical_pressure_values[:,0]
            y2=analytical_pressure_values[:,1]+66.67
        
            fig, ax1, ax2, color1, color2 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$P_{an}$ [MPa]',color1='',y2_txt='$P_{num}$ [MPa]',color2='', title='',mode='2')
        
            plotting_params.object_plot(x2, y2, y1, ax1, ax2, mode='2',color1=color1,color2=color2,label_string='')
            plotting_params.show_plot()           
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_3D_Hydroelasticity_Pcalc_Panal',mode='1')
        
            x1=list(values_time1[:].copy())
            y1=list(values_gen_disp1[:,1].copy())
            x1.insert(0,0)
            y1.insert(0,0)
        
            fig, ax1, color1 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$u_z$ [mm]',color1='',y2_txt='',color2='', title='',mode='1')
        
            plotting_params.object_plot(x1, y1, 'y2', ax1, 'ax2', mode='1',color1=color1,color2='',label_string='')
            plotting_params.show_plot()           
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_3D_Hydroelasticity_u_anal',mode='1')
#
    def test_steady_state_displacement_values(self):
        K=self.my_FEproblem1.mats[0].props[0]
        G=self.my_FEproblem1.mats[0].props[1]
        E=9.*K*G/(K+G)
        M=K+4.*G/3.
        sigma_n= self.my_FEproblem1.Normal_loading_total#Applied normal pressure at the unittests
        pressure_f=self.my_FEproblem1.Pressure_loading
        sigma_eff=sigma_n+pressure_f
        DP= self.my_FEproblem1.DP#Applied pressure at the unittest
        u0=1./K*(DP)*(1./3.) #Measured displacement at the top of the specimen along x0
        u1=1./K*(DP)*(1./3.) #Measured displacement at the top of the specimen along x1    
        u2=1./K*(DP)*(1./3.)*10. #Measured displacement at the top of the specimen along x2    

        self.my_FEproblem1.extract_generalized_force_disp()
        values_time1 = self.my_FEproblem1.array_time
        values_node2_disp0 = self.my_FEproblem1.array_gen_disp[-1,1]
        values_node2_disp1 = self.my_FEproblem1.array_gen_disp[-1,5]
        values_node2_disp2 = self.my_FEproblem1.array_gen_disp[-1,9]

        values_diff=[abs(u0-values_node2_disp0), abs(u1-values_node2_disp1), abs(u2-values_node2_disp2)]
        equal=abs(sum(values_diff))<=1.e-3
        self.assertTrue(equal, "Not identical displacement evolution: "+str(sum(values_diff)))    


if __name__ == '__main__':
    unittest.main()
