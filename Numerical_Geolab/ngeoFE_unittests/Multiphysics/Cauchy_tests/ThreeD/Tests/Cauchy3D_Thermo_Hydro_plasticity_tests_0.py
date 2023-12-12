'''
Created on Mai 30, 2022

@author: Alexandros Stathas

Contains unit tests of ngeoFE applied to Cauchy continua in 3D Thermo-Hydroplasticity with a 
Von-Mises yield criterion. Check pore fluid pressure increase due to thermal
pressurization. The unit cube is under 1D shear.

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
from ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.BVP.CAUCHY_THM_ThermoHydroplasticity_0 import THM3D_FEformulation,THM3D_FEproblem

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

def calculate_dissipation(sigma11,sigma22,sigma33,sigma23,sigma31,sigma12,epsilon11,epsilon22,epsilon33,gamma32,gamma31,gamma12):
    epsilon12=gamma12/2.
    epsilon31=gamma31/2.
    epsilon32=gamma32/2.

    epsilon21=gamma12/2.
    epsilon13=gamma31/2.
    epsilon23=gamma32/2.

    
    W11=sigma11*epsilon11
    W22=sigma22*epsilon22
    W33=sigma33*epsilon33

    W12=sigma12*epsilon12+sigma12*epsilon21
    W13=sigma31*epsilon31+sigma31*epsilon13
    W23=sigma23*epsilon23+sigma23*epsilon32
    
    dissipation= W11+W22+W33+W12+W13+W23
    return dissipation

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
        Tests calculated nodal forces and displacements to values in ./reference_data/Cauchy3D_THM_Thermo_Hydro_plasticity.out
        '''
        self.my_FEproblem1.extract_generalized_force_disp()
        values_time1 = self.my_FEproblem1.array_time
        values_gen_force1 = self.my_FEproblem1.array_gen_force
        values_gen_disp1 = self.my_FEproblem1.array_gen_disp
        
        values1=np.concatenate((values_time1, values_gen_disp1, values_gen_force1), axis=1)
        
        #with open(reference_data_path+"Cauchy3D_THM_Thermo_Hydro_plasticity.out", "wb") as fp:   #Pickling
        #    pickle.dump(values1,fp)        
        
        with open(reference_data_path+"Cauchy3D_THM_Thermo_Hydro_plasticity.out", "rb") as fp:   #Pickling
            values_ref=pickle.load(fp)        
        values_diff=values_ref-values1
    
        equal=abs(np.linalg.norm(values_diff))<=1.e-8 #docker precision is reduced it however better than the accuracy of the equlibrium solver 10e-6 and equal to material solver 10e-8
        self.assertTrue(equal, "Not identical time, displacements, forces: "+str(abs(np.linalg.norm(values_diff))))
        #asserts that data are correct
        #if data are correct then plot diagram
        if equal and activate_plots:
            x1=list(values_time1[:].copy())
            y1=list(values_gen_disp1[:,-1].copy())
            x1.insert(0,0)
            y1.insert(0,0)
            
            x2=list(values_time1[:].copy())
            y2=list(values_gen_disp1[:,-6].copy())
            x2.insert(0,0)
            y2.insert(0,0)
            
            fig, ax1, ax2, color1, color2 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$T\; $[$^o$  C]',color1='',y2_txt='$P$ [MPa]',color2='', title='',mode='2')
            
            plotting_params.object_plot(x1, y1, y2, ax1, ax2, mode='2',color1=color1,color2=color2,label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_3D_Thermo_Hydro_plasticity_T_P',mode='1')
    
            x1=list(values_time1[:].copy())
            y1=list(values_gen_disp1[:,4].copy())
            x1.insert(0,0)
            y1.insert(0,0)

            fig, ax1, color1 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$u_y$ [mm]',color1='',y2_txt='',color2='', title='',mode='1')
            plotting_params.object_plot(x1, y1, y2, ax1, ax2, mode='1',color1=color1,color2='',label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_3D_Thermo_Hydro_plasticity_u_y',mode='1')
            
            x1=list(values_time1[:].copy())
            y1=list(np.sum(values_gen_force1[:,0:4].copy(),axis=1))
            x1.insert(0,0)
            y1.insert(0,0)

            x2=list(values_time1[:].copy())
            y2=list(np.sum(values_gen_force1[:,8:12].copy(),axis=1))
            x2.insert(0,0)
            y2.insert(0,0)

            fig, ax1, ax2, color1, color2 = plotting_params.object_plot_axes('$t$ [s]',y1_txt= '$\sigma_{12}$ [MPa]',color1='',y2_txt='$\sigma_{33}$ [MPa]',color2='', title='',mode='2')
            
            plotting_params.object_plot(x1, y1, y2, ax1, ax2, mode='2',color1=color1,color2=color2, label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_3D_Thermo_Hydro_plasticity_sigma_{12}_sigma_{13}',mode='1')


    def test_final_stress_T_P(self):
        K=self.my_FEproblem1.mats[0].props[0]
        G=self.my_FEproblem1.mats[0].props[1]
        
        self.my_FEproblem1.extract_svars_gauss_point()
        svars=self.my_FEproblem1.array_gp_svars_comp
        self.sigma11=svars[:,0]
        self.sigma22=svars[:,6]
        self.sigma33=svars[:,12]
        self.epsilon11=svars[:,84]
        self.epsilon22=svars[:,90]
        self.epsilon33=svars[:,96]
        a=4*G/3+K
        b=-2*G/3+K
        self.sigma_calc=b*self.epsilon11+b*self.epsilon33+a*self.epsilon33
        
        self.my_FEproblem1.extract_generalized_force_disp()
        values_gen_disp1 = self.my_FEproblem1.array_gen_disp
        self.P=values_gen_disp1[:,-6]
        self.T=values_gen_disp1[:,-1]

        self.sigma_tot=self.sigma_calc-self.P

        self.my_FEproblem1.extract_generalized_force_disp()
        values_gen_force1 = self.my_FEproblem1.array_gen_force       
        self.sigma_tot_num= np.sum(values_gen_force1[:,8:12].copy(),axis=1)

        values_diff=self.sigma_tot.copy()+self.sigma_tot_num.copy()

        equal=abs(sum(values_diff))<=1.e-5
        self.assertTrue(equal, "Not identical total stress an vs num: "+str(sum(values_diff)))    
 
    
    def test_DT_dissipation_equiality(self):
        self.my_FEproblem1.extract_svars_gauss_point()
        svars=self.my_FEproblem1.array_gp_svars_comp
        self.sigma11=svars[:,0]
        self.sigma22=svars[:,6]
        self.sigma33=svars[:,12]
        self.sigma32=svars[:,18]
        self.sigma31=svars[:,24]
        self.sigma12=svars[:,30]
        self.epsilon11=svars[:,36]
        self.epsilon22=svars[:,42]
        self.epsilon33=svars[:,48]
        self.gamma32=svars[:,54]
        self.gamma31=svars[:,60]
        self.gamma12=svars[:,66]
        self.dP=svars[:,72]
        self.dT=svars[:,78]
        
        vfunc = np.vectorize(calculate_dissipation)
        dissipation=vfunc(self.sigma11,self.sigma22,self.sigma33,self.sigma32,self.sigma31,self.sigma12,self.epsilon11,self.epsilon22,self.epsilon33,self.gamma32,self.gamma31,self.gamma12)
        rhoC=self.my_FEproblem1.mats[0].props[6]
        dT_calc=1./rhoC*dissipation

        values_diff=self.dT[1:]-dT_calc[:-1]
        equal=abs(sum(values_diff))<=1.e-8
        self.assertTrue(equal, "Not identical temperature increment: "+str(sum(values_diff)))    
    
    def test_DT_DP_equality(self):
        self.my_FEproblem1.extract_generalized_force_disp()
        values_time1 = self.my_FEproblem1.array_time
        
        self.my_FEproblem1.extract_svars_gauss_point()
        svars=self.my_FEproblem1.array_gp_svars_comp
        self.dP=svars[:,72]
        self.dT=svars[:,78]

        values_diff=self.dT-self.dP
        equal=abs(sum(values_diff))<=1.e-8
        self.assertTrue(equal, "Not identical temperature increment: "+str(sum(values_diff)))    
        if equal and activate_plots:
            x1=list(values_time1[:].copy())
            y1=list(self.dT)
            x1.insert(0,0)
            y1.insert(0,0)
    
            x2=list(values_time1[:].copy())
            y2=list(self.dP)
            x2.insert(0,0)
            y2.insert(0,0)
    
            fig, ax1, ax2, color1, color2 = plotting_params.object_plot_axes('$t$ [s]',y1_txt= '$\Delta T\; $[$^o$  C]',color1='',y2_txt='$\Delta P$ [MPa]',color2='', title='',mode='2')
                
            plotting_params.object_plot(x1, y1, y2, ax1, ax2, mode='2',color1=color1,color2=color2, label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_3D_Thermo_Hydro_plasticity_DT_DP',mode='1')
    
            plotting_params.show_plot()

    
if __name__ == '__main__':
    unittest.main()
