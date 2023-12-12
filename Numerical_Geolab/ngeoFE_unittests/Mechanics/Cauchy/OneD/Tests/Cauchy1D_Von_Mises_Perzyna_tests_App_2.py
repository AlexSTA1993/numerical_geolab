'''
Created on Nov 5, 2018

@author: Alexandros Stathas

Contains unit tests of ngeoFE applied to Cauchy continua in 1D linear elasticity

'''
import unittest
from ngeoFE_unittests.Mechanics.Cauchy.OneD.BVP.Cauchy1D_Von_Mises_Perzyna_App_2 import Cauchy1DFEformulation, Cauchy1DFEproblem

from dolfin import *

from dolfin.cpp.io import HDF5File

import pickle
import numpy as np

from ngeoFE_unittests import ngeo_parameters
from ngeoFE_unittests import plotting_params 

reference_data_path = ngeo_parameters.reference_data_path    

class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''
        Run FE analysis example
        '''
        cls.notfirsttime=True
        cls.my_FEformulation=Cauchy1DFEformulation()
        
        #first slow loading procedurde
        cls.my_FEproblem1=Cauchy1DFEproblem(cls.my_FEformulation)
        cls.my_FEproblem1.give_me_solver_params(scale_t=10e-9)
        cls.converged1=cls.my_FEproblem1.run_analysis_procedure(reference_data_path)   
        
        dsde_data=cls.my_FEproblem1.feobj.dsde2.vector().get_local().reshape((-1,1))
        dsde_values=open(ngeo_parameters.reference_data_path+"P1_dsde_values_App2.text","w")
        
        for row in dsde_data:
            np.savetxt(dsde_values,row)
        
        dsde_values.close()
                
    def test_execution(self):
        '''
        Tests execution and convergence
        '''
        self.assertTrue(self.converged1, "Convergence failed")
    
    def test_shear_force_displacement_values(self):
        '''
        Tests calculated nodal forces and displacements to values in ./reference_data/Cauchy1D_force_disp_values.out
        '''
        self.my_FEproblem1.extract_force_disp()
        values_time1 = self.my_FEproblem1.array_time
        values_force1 = self.my_FEproblem1.array_force
        values_disp1 = self.my_FEproblem1.array_disp
        
        # print(values_time.shape, values_disp.shape, values_force.shape)
        values1=np.concatenate((values_time1, values_disp1, values_force1), axis=1)
        
        with open(reference_data_path+"Cauchy1D_Perzyna_force_disp_values_App_2.out", "wb") as fp:   #Pickling
            pickle.dump(values1,fp)        
        
        with open(reference_data_path+"Cauchy1D_Perzyna_force_disp_values_App_2.out", "rb") as fp:   #Pickling
            values_ref=pickle.load(fp)        
        values_diff=values_ref-values1
    
        equal=abs(np.linalg.norm(values_diff))<=1.e-13
        self.assertTrue(equal, "Not identical time, displacements, forces: "+str(abs(np.linalg.norm(values_diff))))
        #asserts that data are correct
        #if dtat are correct then plot diagram
        if equal:
            x1=list(values_time1[:].copy())
            y1=list(-values_force1[:].copy())
            x1.insert(0,0)
            y1.insert(0,0)

            
            # fig, ax = plotting_params.object_plot_axes('$t$ [s]', '$\sigma$ [kPa]', '')
            # plotting_params.object_plot(x1, y1, ax, '')
            # plotting_params.plot_legends('./reference_data/', fig, ax,legend_title=' ', filename='Cauchy_2D_Perzyna_elastoplastic_App2_s_t',mode='1')
            fig, ax1, color1 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$\sigma$ [MPa]',color1='',y2_txt='',color2='', title='',mode='1')
            plotting_params.object_plot(x1, y1,' y2', ax1, 'ax2', mode='1',color1=color1,color2='',label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_1D_Perzyna_visoplasticity_App_2_sigma_t',mode='1')


    
            x1=list(values_disp1[:].copy())
            y1=list(-values_force1[:].copy())
            x1.insert(0,0)
            y1.insert(0,0)

            
            # fig, ax1 = plotting_params.object_plot_axes('$u$ [mm]', '$\sigma$ [kPa]', '')
            # plotting_params.object_plot(x1, y1, ax1, '')
            # plotting_params.plot_legends('./reference_data/', fig, ax,legend_title=' ', filename='Cauchy_2D_Perzyna_elastoplastic_App2_s_u',mode='1')
            fig, ax1, color1 = plotting_params.object_plot_axes('$u$ [mm]', y1_txt='$\sigma$ [MPa]',color1='',y2_txt='',color2='', title='',mode='1')
            plotting_params.object_plot(x1, y1,' y2', ax1, 'ax2', mode='1',color1=color1,color2='',label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_1D_Perzyna_visoplasticity_App_2_sigma_u',mode='1')

            
            
    # def test_identical_elastoplastic_matrix(self):  
    #     self.my_FEproblem1.extract_force_disp()  
    #     self.my_FEproblem1.extract_elastoplastic_matrix()   
    #     # return
    #     values=self.my_FEproblem1.EH
    #     # print(values)
    #
    #     #write data to binary files
    #     with open(ngeo_parameters.reference_data_path+"Cauchy2D_Perzyna_elastoplastic_modulo_App_2.out", "wb") as fp:   #Pickling
    #         pickle.dump(values, fp)  
    #
    #     #read data from binary files
    #     with open(ngeo_parameters.reference_data_path+"Cauchy2D_Perzyna_elastoplastic_modulo_App_2.out", "rb") as fp:   #Pickling
    #         values_ref=pickle.load(fp)        
    #     values_diff=values_ref-values
    #
    #     equal=abs(np.linalg.norm(values_diff))<=1.e-13
    #     self.assertTrue(equal, "Not identical elastoplastic_moduli: "+str(abs(np.linalg.norm(values_diff))))    
    
    def test_analytical__yield_stress(self):  
        self.my_FEproblem1.extract_force_disp()  
        self.my_FEproblem1.extract_svars_gauss_point()  
        
        gamma_dot_vp_1=np.divide(self.my_FEproblem1.array_gp_svars_comp,self.my_FEproblem1.array_dtime)
        cc=self.my_FEproblem1.mats[-1].props[11]
        etavp=self.my_FEproblem1.mats[-1].props[18]/cc
        tau_yield_anal_1=cc+etavp*cc*gamma_dot_vp_1
       
        values=tau_yield_anal_1

        with open(ngeo_parameters.reference_data_path+"/Cauchy1D_elasto-viscoplastic_stress_App2.out", "wb") as fp:   #Pickling
            pickle.dump(values,fp)    
        
        with open(ngeo_parameters.reference_data_path+"/Cauchy1D_elasto-viscoplastic_stress_App2.out", "rb") as fp:   #Pickling
            values_ref=pickle.load(fp)        
    
        values_diff=values_ref-values
        equal=abs(np.linalg.norm(values_diff))<=1.e-8
        self.assertTrue(equal, "Not identical_analytical_stress: "+str(abs(np.linalg.norm(values_diff))))  

    def test_analytical__yield_stress_comparison(self):  
        self.my_FEproblem1.extract_force_disp()  
        self.my_FEproblem1.extract_svars_gauss_point()  

        gamma_dot_vp_1=np.divide(self.my_FEproblem1.array_gp_svars_comp,self.my_FEproblem1.array_dtime)
        cc=self.my_FEproblem1.mats[-1].props[11]
        etavp=self.my_FEproblem1.mats[-1].props[18]/cc
        tau_yield_anal_1=cc+etavp*cc*gamma_dot_vp_1
        
        values_force1=self.my_FEproblem1.array_force
        diff_values1=values_force1[-1]+tau_yield_anal_1[-1]

        equal=abs(np.linalg.norm(diff_values1))<=1.e-8
        self.assertTrue(equal, "Not identical_analytical_stress_compare_1: "+str(abs(np.linalg.norm(diff_values1))))
         
if __name__ == '__main__':
    unittest.main()