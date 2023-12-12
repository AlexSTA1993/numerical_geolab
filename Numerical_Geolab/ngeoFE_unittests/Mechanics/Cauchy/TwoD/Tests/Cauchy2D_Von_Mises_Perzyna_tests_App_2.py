'''
Created on Nov 5, 2018

@author: Alexandros Stathas

Contains unit tests of ngeoFE applied to Cauchy continua in 1D linear elasticity

'''
import sys
import os
import unittest
from math import *
from ngeoFE_unittests.Mechanics.Cauchy.TwoD.BVP.Cauchy2D_Von_Mises_Perzyna_App_2 import Cauchy2DFEformulation, Cauchy2DFEproblem

from dolfin import *

from dolfin.cpp.io import HDF5File

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
        cls.my_FEformulation=Cauchy2DFEformulation()
        
        #first slow loading procedurde
        cls.my_FEproblem1=Cauchy2DFEproblem(cls.my_FEformulation)
        cls.my_FEproblem1.give_me_solver_params(scale_t=10e-9)
        cls.converged1=cls.my_FEproblem1.run_analysis_procedure(reference_data_path)   
        
#        dsde_data=cls.my_FEproblem1.feobj.dsde2.vector().get_local().reshape((-1,1))
#        dsde_values=open(ngeo_parameters.reference_data_path+"P1_dsde_values_App2.text","w")
        
#        for row in dsde_data:
#            np.savetxt(dsde_values,row)
        
#        dsde_values.close()
        
    def test_execution(self):
        '''
        Tests execution and convergence
        '''
        self.assertTrue(self.converged1, "Convergence failed")
    
    def test_shear_force_displacement_values(self):
        '''
        Tests calculated nodal forces and displacements to values in ./reference_data/Cauchy2D_force_disp_values.out
        '''
        self.my_FEproblem1.extract_force_disp()
        values_time1 = self.my_FEproblem1.array_time
        values_force1 = self.my_FEproblem1.array_force
        values_disp1 = self.my_FEproblem1.array_disp
        
        # print(values_time.shape, values_disp.shape, values_force.shape)
        values1=np.concatenate((values_time1, values_disp1, values_force1), axis=1)
        
        # with open(reference_data_path+"Cauchy2D_Perzyna_force_disp_values_App_2.out", "wb") as fp:   #Pickling
        #     pickle.dump(values1,fp)        
        
        with open(reference_data_path+"Cauchy2D_Perzyna_force_disp_values_App_2.out", "rb") as fp:   #Pickling
            values_ref=pickle.load(fp)        
        values_diff=values_ref-values1
    
        equal=abs(np.linalg.norm(values_diff))<=1.e-7 #docker precision is reduced it however better than the accuracy of the equlibrium solver 10e-6
        self.assertTrue(equal, "Not identical time, displacements, forces: "+str(abs(np.linalg.norm(values_diff))))
        #asserts that data are correct
        #if dtat are correct then plot diagram
        if equal and activate_plots:
            x1=list(values_time1[:].copy())
            y1=list(-values_force1[:].copy())
            x1.insert(0,0)
            y1.insert(0,0)

            fig, ax1, color1 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$\sigma$ [MPa]',color1='',y2_txt='',color2='', title='',mode='1')
            plotting_params.object_plot(x1, y1,' y2', ax1, 'ax2', mode='1',color1=color1,color2='',label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_3D_Perzyna_visoplasticity_App_2_sigma_t',mode='1')
    
            x1=list(values_disp1[:].copy())
            y1=list(-values_force1[:].copy())
            x1.insert(0,0)
            y1.insert(0,0)

            fig, ax1, color1 = plotting_params.object_plot_axes('$u$ [mm]', y1_txt='$\sigma$ [MPa]',color1='',y2_txt='',color2='', title='',mode='1')
            plotting_params.object_plot(x1, y1,' y2', ax1, 'ax2', mode='1',color1=color1,color2='',label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_3D_Perzyna_visoplasticity_App_2_sigma_u',mode='1')

            
            
    def test_analytical__yield_stress2(self):
        dtmin1 = self.my_FEproblem1.slv.dtmin  
        self.my_FEproblem1.extract_force_disp()
        values_time1 = self.my_FEproblem1.array_time
        values_disp1 = self.my_FEproblem1.array_disp

        self.my_FEproblem1.extract_svars_gauss_point()  
        svars_values_1=self.my_FEproblem1.array_gp_svars_comp
        
        values_time1, indices1 = np.unique(values_time1[:,0].copy().round(decimals=-int(floor(log10(abs(dtmin1))))),return_index=True)
        values_dtime1 = self.my_FEproblem1.array_dtime[:,0]

        values_dtime1_unique = values_dtime1[indices1]  

        sigma12_1= svars_values_1[:,10][indices1]

        epsilon12_1 = svars_values_1[:,22][indices1]
        ldot_1=svars_values_1[:,36][indices1]

        depsilon_p_12_1=svars_values_1[:,48][indices1]
        
        gamma_dot_vp_1=np.divide(depsilon_p_12_1,values_dtime1_unique)
        cc=self.my_FEproblem1.mats[-1].props[11]
        etavp=self.my_FEproblem1.mats[-1].props[18]/cc
        tau_yield_anal_1=cc+etavp*cc*gamma_dot_vp_1
    
        
        G = self.my_FEproblem1.mats[-1].props[1]
        gamma_tot = epsilon12_1
        
        tau_star=cc+G*(gamma_tot-cc/G)*np.exp(-G/(cc*etavp)*values_time1)
                
        tau_anal_1_star =np.where(tau_star>cc,tau_star,cc)
        if activate_plots:
            fig, ax1, color1 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$\sigma$ [MPa]',color1='k',y2_txt='',color2='k', title='',mode='1')
            plotting_params.object_plot(values_time1, tau_anal_1_star, sigma12_1[::100], ax1, ax1,values_time1[::100],plot_mode='scatter',color1=['r','b'],color2=['r','b'],label_string='')    
        
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_2D_Perzyna_App_2_Relaxation_scatter',mode='1')
            
    def test_analytical__yield_stress_comparison(self):  
        self.my_FEproblem1.extract_force_disp()  
        self.my_FEproblem1.extract_svars_gauss_point()  
        svars_values_1=self.my_FEproblem1.array_gp_svars_comp
        
        sigma12 = svars_values_1[:,10]
        dgamma_vp_2_1 = svars_values_1[:,48]
                           
        gamma_dot_vp_1=np.divide(dgamma_vp_2_1,self.my_FEproblem1.array_dtime[:,0])
        
        cc=self.my_FEproblem1.mats[-1].props[11]
        etavp=self.my_FEproblem1.mats[-1].props[18]/cc
        tau_yield_anal_1=cc+etavp*cc*gamma_dot_vp_1
        
        values_force1=self.my_FEproblem1.array_force
        
        diff_values1=values_force1[:,0]+tau_yield_anal_1

        equal=abs(np.linalg.norm(diff_values1))<=1.e-6
        self.assertTrue(equal, "Not identical_analytical_stress_compare_1: "+str(abs(np.linalg.norm(diff_values1))))
         
if __name__ == '__main__':
    unittest.main()
