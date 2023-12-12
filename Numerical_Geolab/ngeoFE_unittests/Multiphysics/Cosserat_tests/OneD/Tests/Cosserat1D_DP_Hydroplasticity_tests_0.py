'''
Created on Mai 30, 2022

@author: Alexandros Stathas

Contains unit tests of ngeoFE applied to Cosserat continua in 3D Hydroplasticity with a 
Drucker Prager yield criterion. Check yield limit for different pressure and yielding under isochoric deformation.

Checks:
-Convergence
-Generalised force displacement values
-Yield criterion
'''

import sys
import os
import unittest
from ngeoFE_unittests.Multiphysics.Cosserat_tests.OneD.BVP.Cosserat1D_Drucker_Prager_Hydro_plasticity import CosseratTHM1DFEformulation,CosseratTHM1DFEproblem

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



def calculate_deviatoric_tensor(sigma11,sigma12,sigma13,sigma21,sigma22,sigma23,sigma31,sigma32,sigma33):
    p_tot=(sigma11+sigma22+sigma33)/3.
    s11=sigma11-p_tot
    s12=sigma12
    s13=sigma13
    
    s21=sigma21
    s22=sigma22-p_tot
    s23=sigma23
    
    s31=sigma31
    s32=sigma32
    s33=sigma33-p_tot
        
    sij=np.array(np.zeros((s11.shape[0],3,3)))
    sij[:,0,0]=s11
    sij[:,0,1]=s12
    sij[:,1,0]=s21

    sij[:,1,1]=s22
    sij[:,0,2]=s13
    sij[:,2,0]=s31

    sij[:,2,2]=s33
    sij[:,1,2]=s23
    sij[:,2,1]=s32

    return sij

def tensor_product(sij,sji):
    J2=np.zeros((sij.shape[0]))
    for k in range(sij.shape[0]):
        sum=0.
        for i in range(sij.shape[1]):
            for j in range(sij.shape[2]):
                sum+=sij[k,i,j]*sji[k,i,j]
        J2[k]=sum
    return J2
    
def calculate_J2(sigma11,sigma12,sigma13,sigma21,sigma22,sigma23,sigma31,sigma32,sigma33, h1, h2):
    sij=calculate_deviatoric_tensor(sigma11,sigma12,sigma13,sigma21,sigma22,sigma23,sigma31,sigma32,sigma33)
    
    sji=np.array(np.zeros(sij.shape))
    sji[:]=np.transpose(sij,axes=(0,2,1))
    
    J2_1=h1*tensor_product(sij,sij)
    
    J2_2=h2*tensor_product(sij,sji)
    return J2_1+J2_2

def calculate_eq_tot(epsilon11,epsilon12,epsilon13,epsilon21,epsilon22,epsilon23,epsilon31,epsilon32,epsilon33,g1,g2):
    eij=calculate_deviatoric_tensor(epsilon11,epsilon12,epsilon13,epsilon21,epsilon22,epsilon23,epsilon31,epsilon32,epsilon33)
    
    eji=np.array(np.zeros(eij.shape))
    eji[:]=np.transpose(eij,axes=(0,2,1))
    
    eq_tot_1=g1*tensor_product(eij,eji)
    eq_tot_2=g2*tensor_product(eij,eji)

    return eq_tot_1+eq_tot_2

def assign_generalisezed_stress(svars_values,start=0,step=2,total_comp=9):

    g_svars=[]
    for i in range(start,start+total_comp):
        g_svars.append(svars_values[:,int(i*step)])
        
    return g_svars[0], g_svars[1], g_svars[2], g_svars[3], g_svars[4], g_svars[5], g_svars[6], g_svars[7], g_svars[8] 

def apply_total_gen_strain_rate(t,dot_g11,dot_g12,dot_g13,dot_g21,dot_g22,dot_g23,dot_g31,dot_g32,dot_g33):
    dot_g_t=np.zeros((t.shape[0],3,3))
    dot_g_t[:,0,0]=dot_g11
    dot_g_t[:,0,1]=dot_g12
    dot_g_t[:,0,2]=dot_g13
    dot_g_t[:,1,0]=dot_g21
    dot_g_t[:,1,1]=dot_g22
    dot_g_t[:,1,2]=dot_g23
    dot_g_t[:,2,0]=dot_g31
    dot_g_t[:,2,1]=dot_g32
    dot_g_t[:,2,2]=dot_g33
    return dot_g_t

def assign_material_parameters(mats_param):
    K=mats_param.props[0]
    G=mats_param.props[1]
    Gc=mats_param.props[2]
    
    M=mats_param.props[4]
    Mc=mats_param.props[5]
    
    R=mats_param.props[9]
    tanfi=mats_param.props[10]
    cc=mats_param.props[11]
    tanpsi=mats_param.props[12]
    Hsfi=mats_param.props[13]
    Hscc=mats_param.props[14]

    h1=mats_param.props[15]
    h2=mats_param.props[16]
    h3=mats_param.props[17]
    h4=mats_param.props[18]
    return K, G, Gc, M, Mc, R, tanfi, cc, tanpsi, Hsfi, Hscc, h1, h2, h3, h4

def betakl(K,tanfi,G,Gc,h1,h2,tau,sij):
    newaxis =np.newaxis
    sji=np.array(np.zeros(sij.shape))
    sji[:]=np.transpose(sij,axes=(0,2,1))
    bkl=sij[:]/tau[:,newaxis, newaxis]*((G+Gc)*h1+(G-Gc)*h2)+sji[:]/tau[:,newaxis, newaxis]*((G+Gc)*h2+(G-Gc)*h1)
    bkl[:,0,0]=bkl[:,0,0]+K*tanfi
    bkl[:,1,1]=bkl[:,1,1]+K*tanfi
    bkl[:,2,2]=bkl[:,2,2]+K*tanfi
    return bkl

def calculate_Hp(K,G,Gc,tanfi,tanpsi,sij,tau,h1,h2):
    sijsij=tensor_product(sij,sij)
    A=sijsij/tau**2*((G+Gc)*(h1**2+h2**2)+2*(G-Gc)*h1*h2)
    
    sji=np.array(np.zeros(sij.shape))
    sji[:]=np.transpose(sij,axes=(0,2,1))
    sijsji=tensor_product(sij,sji)
    B=sijsji/tau**2*((G-Gc)*(h1**2+h2**2)+2*(G+Gc)*h1*h2)
    
    Hp=A+B+K*tanfi*tanpsi
    return Hp

class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''
        Run FE analysis example
        '''
        cls.notfirsttime=True
        cls.my_FEformulation=CosseratTHM1DFEformulation()
        
        #first slow loading procedurde
        cls.my_FEproblem1=CosseratTHM1DFEproblem(cls.my_FEformulation,PF=66.66)
        cls.my_FEproblem1.give_me_solver_params(scale_t=1.)
        cls.converged1=cls.my_FEproblem1.run_analysis_procedure(reference_data_path)
        
        cls.my_FEproblem2=CosseratTHM1DFEproblem(cls.my_FEformulation,PF=133.32)
        cls.my_FEproblem2.give_me_solver_params(scale_t=1.)
        cls.converged1=cls.my_FEproblem2.run_analysis_procedure(reference_data_path)            
        
    def test_execution(self):
        '''
        Tests execution and convergence
        '''
        self.assertTrue(self.converged1, "Convergence failed")
    
    def test_generalized_force_displacement_values(self):
        '''
        Tests calculated nodal forces and displacements to values in ./reference_data/Cosserat1D_force_disp_values.out
        '''
        self.my_FEproblem1.extract_generalized_force_disp()
        values_time1 = self.my_FEproblem1.array_time
        values_gen_force1 = self.my_FEproblem1.array_gen_force
        values_gen_disp1 = self.my_FEproblem1.array_gen_disp
        
        values1=np.concatenate((values_time1, values_gen_disp1, values_gen_force1), axis=1)
    
        # with open(reference_data_path+"Cosserat1D_THM_Hydroplasticity.out", "wb") as fp:   #Pickling
        #     pickle.dump(values1,fp)        
    
        with open(reference_data_path+"Cosserat1D_THM_Hydroplasticity.out", "rb") as fp:   #Pickling
            values_ref=pickle.load(fp)        
        values_diff_1=values_ref-values1
    
        self.my_FEproblem1.extract_svars_gauss_point()
        svars_values_1=self.my_FEproblem1.array_gp_svars_comp
        
        self.my_FEproblem2.extract_svars_gauss_point()
        svars_values_2=self.my_FEproblem2.array_gp_svars_comp


        # with open(reference_data_path+"Cosserat1D_THM_Hydroplasticity_svars_values_App_1.out", "wb") as fp:   #Pickling
        #     pickle.dump(svars_values_1,fp)        
    
        with open(reference_data_path+"Cosserat1D_THM_Hydroplasticity_svars_values_App_1.out", "rb") as fp:   #Pickling
            svars_values_ref=pickle.load(fp)        
        values_diff_2=svars_values_ref-svars_values_1

        
        equal_1=abs(np.linalg.norm(values_diff_1))<=1.e-9 #docker precision is reduced it however better than the accuracy of the equlibrium solver 10e-6 and material solver 10e-8
        equal_2=abs(np.linalg.norm(values_diff_2))<=1.e-9 #docker precision is reduced it however better than the accuracy of the equlibrium solver 10e-6 and material solver 10e-8
        self.assertTrue(equal_1 and equal_2, "Not identical time, displacements, forces, svars: "+str(abs(np.linalg.norm(values_diff_1)))+str(abs(np.linalg.norm(values_diff_2))))
        #asserts that data are correct
        #if data are correct then plot diagram
        if equal_1 and equal_2 and activate_plots:
           
            K, G, Gc, M, Mc, R,tanfi,cc,tanpsi,Hsfi,Hscc,h1,h2,h3,h4=assign_material_parameters(self.my_FEproblem1.mats[0])

            sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33=assign_generalisezed_stress(svars_values_1,start=0,step=1,total_comp=9)
            sij= calculate_deviatoric_tensor(sigma11,sigma12,sigma13,sigma21,sigma22,sigma23,sigma31,sigma32,sigma33)

            mu11, mu12, mu13, mu21, mu22, mu23, mu31, mu32, mu33=assign_generalisezed_stress(svars_values_1,start=9,step=1,total_comp=9)
            mij= calculate_deviatoric_tensor(mu11, mu12, mu13, mu21, mu22, mu23, mu31, mu32, mu33)

            epsilon11, epsilon12, epsilon13, epsilon21, epsilon22, epsilon23, epsilon31, epsilon32, epsilon33=assign_generalisezed_stress(svars_values_1,start=24,step=1,total_comp=9)
            kappa11, kappa12, kappa13, kappa21, kappa22, kappa23, kappa31, kappa32, kappa33=assign_generalisezed_stress(svars_values_1,start=33,step=1,total_comp=9)
            
            g1=self.my_FEproblem1.mats[0].props[19]
            g2=self.my_FEproblem1.mats[0].props[20]
            g3=self.my_FEproblem1.mats[0].props[21]
            g4=self.my_FEproblem1.mats[0].props[22]
            

            J2_1=calculate_J2(sigma11,sigma12,sigma13,sigma21,sigma22,sigma23,sigma31,sigma32,sigma33,h1,h2)
            J2_2=calculate_J2(mu11,mu12,mu13,mu21,mu22,mu23,mu31,mu32,mu33,h3,h4)
            
            J2_1_f=np.sqrt(J2_1+1/R**2*J2_2)

            eq_tot_1=calculate_eq_tot(epsilon11,epsilon12,epsilon13,epsilon21,epsilon22,epsilon23,epsilon31,epsilon32,epsilon33,g1,g2)
            eq_tot_2=calculate_eq_tot(kappa11,kappa12,kappa13,kappa21,kappa22,kappa23,kappa31,kappa32,kappa33,g3,g4)
            eq_tot=np.sqrt(eq_tot_1+R**2*eq_tot_2)
            
            p_eff=(sigma11+sigma22+sigma33)/3.
            evol_tot_1=epsilon11+epsilon22+epsilon33           

            sigma11_2, sigma12_2, sigma13_2, sigma21_2, sigma22_2, sigma23_2, sigma31_2, sigma32_2, sigma33_2=assign_generalisezed_stress(svars_values_2,start=0,step=1,total_comp=9)
            sij_2= calculate_deviatoric_tensor(sigma11_2,sigma12_2,sigma13_2,sigma21_2,sigma22_2,sigma23_2,sigma31_2,sigma32_2,sigma33_2)

            mu11_2, mu12_2, mu13_2, mu21_2, mu22_2, mu23_2, mu31_2, mu32_2, mu33_2=assign_generalisezed_stress(svars_values_2,start=9,step=1,total_comp=9)
            mij_2= calculate_deviatoric_tensor(mu11_2, mu12_2, mu13_2, mu21_2, mu22_2, mu23_2, mu31_2, mu32_2, mu33_2)

            epsilon11, epsilon12, epsilon13, epsilon21, epsilon22, epsilon23, epsilon31, epsilon32, epsilon33=assign_generalisezed_stress(svars_values_2,start=24,step=1,total_comp=9)
            kappa11, kappa12, kappa13, kappa21, kappa22, kappa23, kappa31, kappa32, kappa33=assign_generalisezed_stress(svars_values_2,start=33,step=1,total_comp=9)

            J2_1=calculate_J2(sigma11_2, sigma12_2, sigma13_2, sigma21_2, sigma22_2, sigma23_2, sigma31_2, sigma32_2, sigma33_2,h1,h2)
            J2_2=calculate_J2(mu11_2, mu12_2, mu13_2, mu21_2, mu22_2, mu23_2, mu31_2, mu32_2, mu33_2,h3,h4)
            
            J2_2_f=np.sqrt(J2_1+1/R**2*J2_2)

            
            x1=list((sigma11+sigma22+sigma33)/3.)
            y1=J2_1_f.tolist()
            fig, ax1, color1 = plotting_params.object_plot_axes('$p^\prime$ [MPa]', y1_txt='$\sqrt{J_2}$ [MPa]',color1='',y2_txt='',color2='', title='',mode='1')
    
            plotting_params.object_plot(x1, y1, "y1", ax1, 'ax2', mode='1',color1=color1,color2='color2',label_string='')
            plotting_params.show_plot()           
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cosserat_1D_Hydroplasticity_0_J2_P',mode='1')
        
            x1=list(values_time1[:].copy())
            y1=list(epsilon11+epsilon22+epsilon33)
            x1.insert(0,0)
            y1.insert(0,0)
    
            fig, ax1, color1 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$\\varepsilon_{v}$',color1='',y2_txt='',color2='', title='',mode='1')
            plotting_params.object_plot(x1, y1, "y1", ax1, 'ax2', mode='1',color1=color1,color2='color2',label_string='')
            plotting_params.show_plot()           
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cosserat_1D_Hydroplasticity_0_evol',mode='1')
    
            x1=list(values_time1[:].copy())
            y1=J2_1_f.tolist()
            x1.insert(0,0)
            y1.insert(0,0)
    
            x2=list(values_time1[:].copy())
            y2=J2_2_f.tolist()
            x2.insert(0,0)
            y2.insert(0,0)
    
            x3=list(values_time1[:].copy())
            y3=J2_2_f.tolist()
            x3.insert(0,0)
            y3.insert(0,0)
    
    
            fig, ax1, color1 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$\sqrt{J_2}\; $[MPa]',color1='',y2_txt='$J^2_2\; $[MPa]',color2='', title='',mode='1')
            plotting_params.object_plot(x2, y2, y1, ax1, 'ax2',x1, mode='4',color1=color1,color2='',label_string='')
            plotting_params.show_plot()                       
    
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cosserat_1D_Hydroplasticity_P1_vs_P2',mode='1')
    
            plotting_params.show_plot()
    
    
    def check_yielding_criterion(self):
        self.my_FEproblem1.extract_svars_gauss_point()
        svars_values_1=self.my_FEproblem1.array_gp_svars_comp
        
        K, G, Gc, M, Mc, R,tanfi,cc,tanpsi,Hsfi,Hscc,h1,h2,h3,h4=assign_material_parameters(self.my_FEproblem1.mats[0])

        sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33=assign_generalisezed_stress(svars_values_1,start=0,step=1,total_comp=9)
        sij= calculate_deviatoric_tensor(sigma11,sigma12,sigma13,sigma21,sigma22,sigma23,sigma31,sigma32,sigma33)

        mu11, mu12, mu13, mu21, mu22, mu23, mu31, mu32, mu33=assign_generalisezed_stress(svars_values_1,start=9,step=1,total_comp=9)
        mij= calculate_deviatoric_tensor(mu11, mu12, mu13, mu21, mu22, mu23, mu31, mu32, mu33)

        epsilon11, epsilon12, epsilon13, epsilon21, epsilon22, epsilon23, epsilon31, epsilon32, epsilon33=assign_generalisezed_stress(svars_values_1,start=24,step=1,total_comp=9)
        kappa11, kappa12, kappa13, kappa21, kappa22, kappa23, kappa31, kappa32, kappa33=assign_generalisezed_stress(svars_values_1,start=33,step=1,total_comp=9)
            
        g1=self.my_FEproblem1.mats[0].props[19]
        g2=self.my_FEproblem1.mats[0].props[20]
        g3=self.my_FEproblem1.mats[0].props[21]
        g4=self.my_FEproblem1.mats[0].props[22]
            

        J2_1=calculate_J2(sigma11,sigma12,sigma13,sigma21,sigma22,sigma23,sigma31,sigma32,sigma33,h1,h2)
        J2_2=calculate_J2(mu11,mu12,mu13,mu21,mu22,mu23,mu31,mu32,mu33,h3,h4)
            
        J2_1_f=np.sqrt(J2_1+1/R**2*J2_2)
        
        p_tot=(sigma11+sigma22+sigma33)/3.
        
        tanfi=self.my_FEproblem1.mats[0].props[10]
        F=J2_1_f-tanfi*p_tot
        equal=abs(np.linalg.norm(F))<=1.e-9 #docker precision is reduced it however better than the accuracy of the equlibrium solver 10e-6 and material solver 10e-8
        self.assertTrue(equal, "yield criterion not respected: "+str(abs(np.linalg.norm(F))))
        

if __name__ == '__main__':
    unittest.main()
