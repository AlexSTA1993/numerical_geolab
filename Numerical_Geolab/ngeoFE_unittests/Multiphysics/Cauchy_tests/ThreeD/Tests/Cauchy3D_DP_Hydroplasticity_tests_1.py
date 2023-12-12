'''
Created on Mai 30, 2022

@author: Alexandros Stathas

Contains unit tests of ngeoFE applied to Cauchy continua in 3D Hydroplasticity with a 
Drucker Prager yield criterion. Check softening behavior due to pore fluid pressure increase. The unit cube is under 1D compression.

Contains unit tests of ngeoFE applied to Cauchy continua in 3D Hydroelasticity.
Checks:
-Convergence
-Generalized force displacement values
'''
import sys
import os
import unittest
from ngeoFE_unittests.Multiphysics.Cauchy_tests.ThreeD.BVP.CAUCHY_DP_HM_Hydroplasticity_1 import THM3D_FEformulation,THM3D_FEproblem

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

def product_sij_sji(sij,sji):
    # J2=sij[:,0,0]*sij[:,0,0]+sij[:,1,1]*sij[:,1,1]+sij[:,2,2]*sij[:,2,2]
    J2=np.zeros((sij.shape[0]))
    for k in range(sij.shape[0]):
        sum=0.
        for i in range(3):
            for j in range(3):
                sum+=sij[k,i,j]*sji[k,i,j]
        J2[k]=sum
    return J2
    
def calculate_J2(sigma11,sigma22,sigma33,sigma23,sigma13,sigma12):
    p_tot=(sigma11+sigma22+sigma33)/3.
    s11=sigma11-p_tot
    s22=sigma22-p_tot
    s33=sigma33-p_tot
    s23=sigma23
    s13=sigma13
    s12=sigma12
        
    sij=np.array(np.zeros((s11.shape[0],3,3)))
    sij[:,0,0]=s11
    sij[:,0,1]=s12
    sij[:,1,0]=-s12

    sij[:,1,1]=s22
    sij[:,0,2]=s13
    sij[:,2,0]=-s13

    sij[:,2,2]=s33
    sij[:,1,2]=s23
    sij[:,2,1]=-s23
    
    sji=np.array(np.zeros(sij.shape))
    sji[:]=np.transpose(sij,axes=(0,2,1))
    
    J2=np.sqrt(1./2.*np.abs(product_sij_sji(sij,sji)))
    return J2

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
        Tests calculated nodal forces and displacements to values in ./../../../reference_data/Cauchy2D_force_disp_values.out
        '''
        self.my_FEproblem1.extract_generalized_force_disp()
        values_time1 = self.my_FEproblem1.array_time
        values_gen_force1 = self.my_FEproblem1.array_gen_force
        values_gen_disp1 = self.my_FEproblem1.array_gen_disp
    
        values1=np.concatenate((values_time1, values_gen_disp1, values_gen_force1), axis=1)
    
        #with open(reference_data_path+"Cauchy3D_THM_Hydroplasticity_2.out", "wb") as fp:   #Pickling
        #    pickle.dump(values1,fp)        
    
        with open(reference_data_path+"Cauchy3D_THM_Hydroplasticity_2.out", "rb") as fp:   #Pickling
            values_ref=pickle.load(fp)        
        values_diff=values_ref-values1
    
        equal=abs(np.linalg.norm(values_diff))<=1.e-9 #docker precision is reduced it however better than the accuracy of the equlibrium solver 10e-6 and material solver 10e-8
        self.assertTrue(equal, "Not identical time, displacements, forces: "+str(abs(np.linalg.norm(values_diff))))
        #asserts that data are correct
        #if data are correct then plot diagram
        if equal and activate_plots:
            sigma11= np.sum(values_gen_force1[:,0:4].copy(),axis=1)
            sigma22= np.sum(values_gen_force1[:,4:8].copy(),axis=1)
            sigma33= np.sum(values_gen_force1[:,8:12].copy(),axis=1)
            sigma23= np.sum(values_gen_force1[:,12:16].copy(),axis=1)
            sigma13= np.sum(values_gen_force1[:,16:20].copy(),axis=1)
            sigma12= np.sum(values_gen_force1[:,20:24].copy(),axis=1)

            J2_1=calculate_J2(sigma11,sigma22,sigma33,sigma23,sigma13,sigma12)
            p_eff=-(sigma11+sigma22+sigma33)/3.+values_gen_disp1[:,-5]
            p=values_gen_disp1[:,-5]
            
            
            x1=p_eff.tolist()
            y1=J2_1.tolist()
            x1.insert(0,self.my_FEproblem1.Normal_loading_eff_1)
            y1.insert(0,0)
            

            fig, ax1, color1 = plotting_params.object_plot_axes('$p^\prime$ [MPa]', y1_txt='$\sqrt{J_2}\;$[MPa]',color1='',y2_txt='',color2='', title='',mode='1')
    
            plotting_params.object_plot(x1, y1, "y1", ax1, 'ax2', mode='1',color1=color1,color2='color2',label_string='')
            plotting_params.show_plot()           
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_3D_Hydroplasticity_2_P',mode='1')
    
    
            x2=list(values_time1[:].copy())
            y2=list(values_gen_disp1[:,3].copy()+values_gen_disp1[:,7].copy()+values_gen_disp1[:,11].copy())
            x2.insert(0,0)
            y2.insert(0,0)
    
    
            fig, ax1, color1 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$\\varepsilon_{v}$',color1='',y2_txt='',color2='', title='',mode='1')
            plotting_params.object_plot(x2, y2, "y1", ax1, 'ax2', mode='1',color1=color1,color2='color2',label_string='')
            plotting_params.show_plot()           
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_3D_Hydroplasticity_2_evol',mode='1')

            x1=p.tolist()
            y1=J2_1.tolist()
            x1.insert(0,self.my_FEproblem1.Pressure_loading)
            y1.insert(0,0)
    
            fig, ax1, ax2, color1, color2 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$P\; $[MPa]',color1='',y2_txt='$\\varepsilon_{v}$',color2='', title='',mode='2')
            
            plotting_params.object_plot(x2, x1, y2, ax1, ax2, mode='2',color1=color1,color2=color2,label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_3D_Hydroplasticity_2_P_evol',mode='1')
if __name__ == '__main__':
    unittest.main()
