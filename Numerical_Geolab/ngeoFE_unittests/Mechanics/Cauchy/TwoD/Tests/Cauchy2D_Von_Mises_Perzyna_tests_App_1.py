'''
Created on Nov 5, 2018

@author: Alexandros Stathas

Contains unit tests of ngeoFE applied to Cauchy continua in 1D linear elasticity

'''
import sys
import os
import unittest
from ngeoFE_unittests.Mechanics.Cauchy.TwoD.BVP.Cauchy2D_Von_Mises_Perzyna_App_1 import Cauchy2DFEformulation, Cauchy2DFEproblem

from dolfin import *

from dolfin.cpp.io import HDF5File

import pickle
import numpy as np

from ngeoFE_unittests import ngeo_parameters
from ngeoFE_unittests import plotting_params 

from math import log10, floor

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
        cls.my_FEproblem1.give_me_solver_params(scale_t=10e-3)
        cls.converged1=cls.my_FEproblem1.run_analysis_procedure(reference_data_path)

        #second fast loading procedure
        cls.my_FEproblem2=Cauchy2DFEproblem(cls.my_FEformulation)
        cls.my_FEproblem2.give_me_solver_params(scale_t=10e-4)
        cls.converged2=cls.my_FEproblem2.run_analysis_procedure(reference_data_path)
        
    def test_execution(self):
        '''
        Tests execution and convergence
        '''
        self.assertTrue(self.converged1, "Convergence failed")
        self.assertTrue(self.converged2, "Convergence failed")

    
    def test_shear_force_displacement_values(self):
        '''
        Tests calculated nodal forces and displacements to values in ./reference_data/Cauchy2D_force_disp_values.out
        '''
        self.my_FEproblem1.extract_force_disp()
        values_time1 = self.my_FEproblem1.array_time
        values_force1 = self.my_FEproblem1.array_force
        values_disp1 = self.my_FEproblem1.array_disp
    
        self.my_FEproblem2.extract_force_disp()        
        values_time2 = self.my_FEproblem2.array_time
        values_force2 = self.my_FEproblem2.array_force
        values_disp2 = self.my_FEproblem2.array_disp
    
        # print(values_time.shape, values_disp.shape, values_force.shape)
        values1=np.concatenate((values_time1, values_disp1, values_force1), axis=1)
        values2=np.concatenate((values_time2, values_disp2, values_force2), axis=1)
        values=np.concatenate((values1,values2))

        # with open(reference_data_path+"/Cauchy2D_Perzyna_force_disp_values_App_1.out", "wb") as fp:   #Pickling
        #     pickle.dump(values, fp)
        #read data from binary files
        with open(reference_data_path+"/Cauchy2D_Perzyna_force_disp_values_App_1.out", "rb") as fp:   #Pickling
            values_ref=pickle.load(fp)        
        values_diff=values_ref-values
                
        equal=abs(np.linalg.norm(values_diff))<=1.e-9 #docker precision is reduced it however better than the accuracy of the equlibrium solver 10e-6 and material solver 10e-8
        self.assertTrue(equal, "Not identical time, displacements, forces: "+str(abs(np.linalg.norm(values_diff))))
        
        #asserts that data are correct
        #if dtat are correct then plot diagram
        if equal and activate_plots:
            x1=list(values_disp1[:].copy())
            y1=list(-values_force1[:].copy())
            x1.insert(0,0)
            y1.insert(0,0)
    
            x2=list(values_disp2[:].copy())
            y2=list(-values_force2[:].copy())
            x2.insert(0,0)
            y2.insert(0,0)
                        
            fig, ax1, color1 = plotting_params.object_plot_axes('$u$ [mm]', y1_txt='$\sigma$ [kPa]',color1='k',y2_txt='',color2='', title='',mode='1')
            
            plotting_params.object_plot(x1, y1, y2, ax1, 'ax2',x2, mode='3',color1='r',color2='b',label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_3D_Perzyna_visoplasticity_App_1_sigma_u_new',mode='1')
    
    def test_analytical__yield_stress(self):
        dtmin1 = self.my_FEproblem1.slv.dtmin  
        self.my_FEproblem1.extract_force_disp()
        values_time1 = self.my_FEproblem1.array_time
        values_disp1 = self.my_FEproblem1.array_disp

        self.my_FEproblem1.extract_svars_gauss_point()  
        svars_values_1=self.my_FEproblem1.array_gp_svars_comp
        
        values_time1, indices1 = np.unique(values_time1[:,0].copy().round(decimals=-int(floor(log10(abs(dtmin1))))),return_index=True)
        values_dtime1 =  self.my_FEproblem1.array_dtime[:,0]
        values_dtime1_unique = values_dtime1[values_dtime1>dtmin1]
        

        
        dtmin2 = self.my_FEproblem2.slv.dtmin        
        self.my_FEproblem2.extract_force_disp()
        values_time2 = self.my_FEproblem2.array_time
        values_disp2 = self.my_FEproblem2.array_disp

        self.my_FEproblem2.extract_svars_gauss_point()  
        svars_values_2=self.my_FEproblem2.array_gp_svars_comp
                
        values_time2, indices2 = np.unique(values_time2[:,0].copy().round(decimals=-int(floor(log10(abs(dtmin2))))),return_index=True)
        values_dtime2 =  self.my_FEproblem2.array_dtime[:,0]
        values_dtime2_unique = values_dtime2[values_dtime2>dtmin2]
        
        sigma12_1= svars_values_1[:,10][indices1]
        sigma12_2= svars_values_2[:,10][indices2]
        epsilon12_1 = svars_values_1[:,22][indices1]
        epsilon12_2 = svars_values_2[:,22][indices2]
        ldot_1=svars_values_1[:,36][indices1]
        ldot_2=svars_values_2[:,36][indices2]
        depsilon_p_12_1=svars_values_1[:,48][indices1]
        depsilon_p_12_2=svars_values_2[:,48][indices2]
        

        gamma_dot_vp_1=np.divide(depsilon_p_12_1,values_dtime1_unique)
        cc=self.my_FEproblem1.mats[-1].props[11]
        etavp=self.my_FEproblem1.mats[-1].props[18]/cc
        tau_yield_anal_1=cc+etavp*cc*gamma_dot_vp_1
    
        self.my_FEproblem2.extract_force_disp()  
        self.my_FEproblem2.extract_elastoplastic_matrix()  
        self.my_FEproblem2.extract_svars_gauss_point()  
    
        gamma_dot_vp_2=np.divide(depsilon_p_12_2,values_dtime2_unique)
        cc=self.my_FEproblem2.mats[-1].props[11]
        etavp=self.my_FEproblem2.mats[-1].props[18]/cc
        tau_yield_anal_2=cc+etavp*cc*gamma_dot_vp_2
        
        G = self.my_FEproblem1.mats[-1].props[1]
        gamma_tot = epsilon12_1
        tau_el_1 = G * gamma_tot 
        tau_anal_1_star_el = np.where(gamma_dot_vp_1<10**(-8),tau_el_1,0)
        
        G = self.my_FEproblem2.mats[-1].props[1]
        gamma_tot = epsilon12_2
        tau_el_2 = G * gamma_tot 
        tau_anal_2_star_el = np.where(gamma_dot_vp_2<10**(-8),tau_el_2,0)
        
        tau_anal_1_star=tau_anal_1_star_el+np.where(gamma_dot_vp_1>0,tau_yield_anal_1,0)
        tau_anal_2_star=tau_anal_2_star_el+np.where(gamma_dot_vp_2>0,tau_yield_anal_2,0)
        
        if activate_plots:        
            fig, ax1, color1 = plotting_params.object_plot_axes('$u$ [mm]', y1_txt='$\sigma$ [MPa]',color1='k',y2_txt='$\sigma_{comp}$ [MPa]',color2='k', title='',mode='1')
            plotting_params.object_plot_doule(ax1,epsilon12_2,sigma12_1,sigma12_2,ax1,epsilon12_2,tau_anal_1_star,tau_anal_2_star, mode='2',color1=['r','b'],color2=['r','b'],label_string='')    
        
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_3D_Perzyna_visoplasticity_App_1_sigma_u_new_an',mode='1')
        
        values=np.concatenate((tau_anal_1_star,tau_anal_2_star))
        #
        # with open(reference_data_path+"/Cauchy2D_elasto-viscoplastic_stress_App1.out", "wb") as fp:   #Pickling
        #     pickle.dump(values,fp)    
    
        with open(reference_data_path+"/Cauchy2D_elasto-viscoplastic_stress_App1.out", "rb") as fp:   #Pickling
            values_ref=pickle.load(fp)        
    
        values_diff=values_ref-values
        equal=abs(np.linalg.norm(values_diff))<=1.e-8
        self.assertTrue(equal, "Not identical_analytical_stress: "+str(abs(np.linalg.norm(values_diff))))  
    
    def test_analytical__yield_stress_comparison(self):  
        dtmin1 = self.my_FEproblem1.slv.dtmin  
        self.my_FEproblem1.extract_force_disp()
        values_time1 = self.my_FEproblem1.array_time
        values_disp1 = self.my_FEproblem1.array_disp

        self.my_FEproblem1.extract_svars_gauss_point()  
        svars_values_1=self.my_FEproblem1.array_gp_svars_comp
        
        values_time1, indices1 = np.unique(values_time1[:,0].copy().round(decimals=-int(floor(log10(abs(dtmin1))))),return_index=True)
        values_dtime1 =  self.my_FEproblem1.array_dtime[:,0]
        values_dtime1_unique = values_dtime1[values_dtime1>dtmin1]
                
        sigma12_1= svars_values_1[:,10][indices1]
        epsilon12_1 = svars_values_1[:,22][indices1]
        ldot_1=svars_values_1[:,36][indices1]
        depsilon_p_12_1=svars_values_1[:,48][indices1]
        gamma_dot_vp_1=np.divide(depsilon_p_12_1,values_dtime1_unique)
        cc=self.my_FEproblem1.mats[-1].props[11]
        etavp=self.my_FEproblem1.mats[-1].props[18]/cc
        tau_yield_anal_1=cc+etavp*cc*gamma_dot_vp_1
    
        values_force1=self.my_FEproblem1.array_force[indices1]
        
        diff_values1=values_force1[-1,0]+tau_yield_anal_1[-1]

        equal=abs(np.linalg.norm(diff_values1))<=1.e-8
        self.assertTrue(equal, "Not identical_analytical_stress_compare_1: "+str(abs(np.linalg.norm(diff_values1))))
    
    def test_analytical__yield_stress_comparison_2(self):  
        dtmin2 = self.my_FEproblem2.slv.dtmin        
        self.my_FEproblem2.extract_force_disp()
        values_time2 = self.my_FEproblem2.array_time
        values_disp2 = self.my_FEproblem2.array_disp

        self.my_FEproblem2.extract_svars_gauss_point()  
        svars_values_2=self.my_FEproblem2.array_gp_svars_comp
                
        values_time2, indices2 = np.unique(values_time2[:,0].copy().round(decimals=-int(floor(log10(abs(dtmin2))))),return_index=True)
        values_dtime2 =  self.my_FEproblem2.array_dtime[:,0]
        values_dtime2_unique = values_dtime2[values_dtime2>dtmin2]
        
        sigma12_2= svars_values_2[:,10][indices2]
        epsilon12_2 = svars_values_2[:,22][indices2]
        ldot_2=svars_values_2[:,36][indices2]
        depsilon_p_12_2=svars_values_2[:,48][indices2]
            
        self.my_FEproblem2.extract_force_disp()  
        self.my_FEproblem2.extract_elastoplastic_matrix()  
        self.my_FEproblem2.extract_svars_gauss_point()  
    
        gamma_dot_vp_2=np.divide(depsilon_p_12_2,values_dtime2_unique)
        cc=self.my_FEproblem2.mats[-1].props[11]
        etavp=self.my_FEproblem2.mats[-1].props[18]/cc
        tau_yield_anal_2=cc+etavp*cc*gamma_dot_vp_2
        
        values_force2=self.my_FEproblem2.array_force[indices2]
        diff_values2=values_force2[-1,0]+tau_yield_anal_2[-1]
        equal=abs(np.linalg.norm(diff_values2))<=1.e-8
        self.assertTrue(equal, "Not identical_analytical_stress_compare_2: "+str(abs(np.linalg.norm(diff_values2))))
        
if __name__ == '__main__':
    unittest.main()
