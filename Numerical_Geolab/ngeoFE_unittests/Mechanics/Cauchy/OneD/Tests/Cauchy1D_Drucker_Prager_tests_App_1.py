'''
Created on Nov 5, 2018

@author: Alexandros Stathas

Contains unit tests of ngeoFE applied to Cauchy continua in 1D linear elasticity

'''
import os
import unittest
from ngeoFE_unittests.Mechanics.Cauchy.OneD.BVP.Cauchy1D_Drucker_Prager_App_1 import Cauchy1DFEformulation, Cauchy1DFEproblem

from dolfin import *

from dolfin.cpp.io import HDF5File

import pickle
import numpy as np

from ngeoFE_unittests import ngeo_parameters
from ngeoFE_unittests import plotting_params 

reference_data_path = ngeo_parameters.reference_data_path    

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
    s11=sigma11-p_tot#+PF
    s22=sigma22-p_tot#+PF
    s33=sigma33-p_tot#+PF
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

def product_eij_eji(eij,eji):
    eq=np.zeros((eij.shape[0]))
    for k in range(eij.shape[0]):
        sum=0.
        for i in range(3):
            for j in range(3):
                sum+=eij[k,i,j]*eji[k,i,j]
        eq[k]=sum
    return eq


def calculate_eq_tot(epsilon11,epsilon22,epsilon33,epsilon23,epsilon13,epsilon12):
    evol_tot=(epsilon11+epsilon22+epsilon33)
    e11=epsilon11-evol_tot/3.#+PF
    e22=epsilon22-evol_tot/3.#+PF
    e33=epsilon33-evol_tot/3.#+PF
    e23=epsilon23
    e13=epsilon13
    e12=epsilon12
        
    eij=np.array(np.zeros((e11.shape[0],3,3)))
    eij[:,0,0]=e11
    eij[:,0,1]=e12
    eij[:,1,0]=-e12

    eij[:,1,1]=e22
    eij[:,0,2]=e13
    eij[:,2,0]=-e13

    eij[:,2,2]=e33
    eij[:,1,2]=e23
    eij[:,2,1]=-e23
    
    eji=np.array(np.zeros(eij.shape))
    eji[:]=np.transpose(eij,axes=(0,2,1))
    
    eq_tot=np.sqrt(2.*np.abs(product_eij_eji(eij,eji)))
    return eq_tot

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
        cls.my_FEproblem1.give_me_solver_params(scale_t=1.)
        cls.converged1=cls.my_FEproblem1.run_analysis_procedure(reference_data_path)       
        # if cls.converged==True: cls.my_FEproblem.plot_me()
        
    def test_execution(self):
        '''
        Tests execution and convergence
        '''
        self.assertTrue(self.converged1, "Convergence failed")
    
    def test_shear_force_displacement_values(self):
        '''
        Tests calculated nodal forces and displacements to values in ./reference_data/Cauchy1D_Drucker_Prager_force_disp_values.out
        '''
        self.my_FEproblem1.extract_force_disp()
        values_time1 = self.my_FEproblem1.array_time
        values_force1 = self.my_FEproblem1.array_force
        values_disp1 = self.my_FEproblem1.array_disp

        
        print(values_time1.shape, values_disp1.shape, values_force1.shape)
        values1=np.concatenate((values_time1, values_disp1, values_force1), axis=1)

        with open(reference_data_path+"Cauchy1D_Drucker_Prager_force_disp_values_App_1.out", "wb") as fp:   #Pickling
            pickle.dump(values1,fp)        
        

        with open(reference_data_path+"Cauchy1D_Drucker_Prager_force_disp_values_App_1.out", "rb") as fp:   #Pickling
            values_ref=pickle.load(fp)        
        values_diff_1=values_ref-values1
 
        self.my_FEproblem1.extract_svars_gauss_point()
        svars_values=self.my_FEproblem1.array_gp_svars_comp
        
        with open(reference_data_path+"Cauchy1D_Drucker_Prager_svars_values_App_1.out", "wb") as fp:   #Pickling
            pickle.dump(svars_values,fp)        
    
        with open(reference_data_path+"Cauchy1D_Drucker_Prager_svars_values_App_1.out", "rb") as fp:   #Pickling
            svars_values_ref=pickle.load(fp)        
        values_diff_2=svars_values_ref-svars_values
  
    
        equal_1=abs(np.linalg.norm(values_diff_1))<=1.e-13
        equal_2=abs(np.linalg.norm(values_diff_2))<=1.e-13
        self.assertTrue(equal_1 and equal_2, "Not identical time, displacements, forces, svars: "+str(abs(np.linalg.norm(values_diff_1)))+str(abs(np.linalg.norm(values_diff_2))))
        #asserts that data are correct
        #if dtat are correct then plot diagram
        if equal_1 and equal_2:
            ny=self.my_FEproblem1.ny
            # nw=self.my_FEproblem1.nw
            
            sigma11= svars_values[:,0]
            sigma22= svars_values[:,2]
            sigma33= svars_values[:,4]
            sigma23= svars_values[:,6]
            sigma13= svars_values[:,8]
            sigma12= svars_values[:,10]
            #
            epsilon11= svars_values[:,12]
            epsilon22= svars_values[:,14]
            epsilon33= svars_values[:,16]
            epsilon23= svars_values[:,18]
            epsilon13= svars_values[:,20]
            epsilon12= svars_values[:,22]            
            
            J2_1=calculate_J2(sigma11,sigma22,sigma33,sigma23,sigma13,sigma12)
            eq_tot_1=calculate_eq_tot(epsilon11,epsilon22,epsilon33,epsilon23,epsilon13,epsilon12)

            p_eff=(sigma11+sigma22+sigma33)/3.
            evol_tot_1=epsilon11+epsilon22+epsilon33
            

            
            
            x1=list(values_time1[:].copy())
            y1=list(J2_1.copy())
            x1.insert(0,0)
            y1.insert(0,0.)
            
            x2=list(values_time1[:].copy())
            y2=list(evol_tot_1.copy())
            x2.insert(0,0)
            y2.insert(0,0.)
            
            x3=list(p_eff.copy())
            x3.insert(0,p_eff[0])


        # if equal:
        #     x1=list(values_time1[:].copy())
        #     y1=list(-values_force1[:].copy())
        #     x1.insert(0,0)
        #     y1.insert(0,0)
    
            fig, ax1, color1 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$\sqrt{J_2}$ [MPa]',color1='k',y2_txt='',color2='', title='',mode='1')
            plotting_params.object_plot(x1, y1,'y2', ax1, 'ax2','x2', mode='1',color1=color1,color2='',label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_1D_Drucker-Prager_App_1_J2_t',mode='1')
            
            x1=list(eq_tot_1.copy())
            y1=list(J2_1.copy())
            x1.insert(0,0)
            y1.insert(0,0)
        
            fig, ax1, color1 = plotting_params.object_plot_axes('$q^{tot}$', y1_txt='$\sqrt{J_2}$ [MPa]',color1='k',y2_txt='',color2='', title='',mode='1')
            plotting_params.object_plot(x1, y1,'y2', ax1, 'ax2','x2', mode='1',color1='r',color2='b',label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_1D_Drucker-Prager_App_1_J2_q',mode='1')
            
            fig, ax1, color1 = plotting_params.object_plot_axes('$p$ [MPa]', y1_txt='$\sqrt{J_2}$ [MPa]',color1='k',y2_txt='',color2='', title='',mode='1')
            plotting_params.object_plot(x3, y1,'y2', ax1, 'ax2','x2', mode='1',color1='r',color2='b',label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cauchy_1D_Drucker-Prager_App_1_J2_p',mode='1')
    
    def test_identical_elastoplastic_matrix(self):  
        self.my_FEproblem1.extract_force_disp()  
        self.my_FEproblem1.extract_elastoplastic_matrix()   
        # return
        values=self.my_FEproblem1.EH
        # print(values)
    
        #write data to binary files
        with open(reference_data_path+"Cauchy1D_Drucker_Prager_elastoplastic_modulo_App_1.out", "wb") as fp:   #Pickling
            pickle.dump(values, fp)  
    
        #read data from binary files
        with open(reference_data_path+"Cauchy1D_Drucker_Prager_elastoplastic_modulo_App_1.out", "rb") as fp:   #Pickling
            values_ref=pickle.load(fp)        
        values_diff=values_ref-values
    
        equal=abs(np.linalg.norm(values_diff))<=1.e-13
        self.assertTrue(equal, "Not identical elastoplastic_moduli: "+str(abs(np.linalg.norm(values_diff))))    
    
    def test_analytical__lambda_dot(self):  
        self.my_FEproblem1.extract_svars_gauss_point()  
        svars_values=self.my_FEproblem1.array_gp_svars_comp

        sigma11= svars_values[:,0]
        sigma22= svars_values[:,2]
        sigma33= svars_values[:,4]
        sigma23= svars_values[:,6]
        sigma13= svars_values[:,8]
        sigma12= svars_values[:,10]
        
        epsilon_p_11=svars_values[:,24]
        epsilon_p_22=svars_values[:,26]
        epsilon_p_33=svars_values[:,28]
        epsilon_p_23=svars_values[:,30]
        epsilon_p_13=svars_values[:,32]
        epsilon_p_12=svars_values[:,34]
                
        p_eff=(sigma11+sigma22+sigma33)/3.
        
        u1=0.02449428
        dot_epsilon_tot11= u1/10.
        dot_epsilon_tot22= -2*u1/10.
        dot_epsilon_tot33= 0.0
        dot_epsilon_tot23= 0.0
        dot_epsilon_tot13= 0.0
        dot_epsilon_tot12= 0.0
        
        dot_evol_tot=dot_epsilon_tot11+dot_epsilon_tot22+dot_epsilon_tot33
        
        s11=sigma11-p_eff
        s22=sigma22-p_eff
        s33=sigma33-p_eff
        s23=sigma23
        s13=sigma13
        s12=sigma12
        
        J2_1=calculate_J2(s11,s22,s33,s23,s13,s12)
        eq_p=calculate_eq_tot(epsilon_p_11,epsilon_p_22,epsilon_p_33,epsilon_p_23,epsilon_p_13,epsilon_p_12)
        
        dot_e_tot11=dot_epsilon_tot11-dot_evol_tot/3.
        dot_e_tot22=dot_epsilon_tot22-dot_evol_tot/3.
        dot_e_tot33=dot_epsilon_tot33-dot_evol_tot/3.
        dot_e_tot23=dot_epsilon_tot23
        dot_e_tot13=dot_epsilon_tot13
        dot_e_tot12=dot_epsilon_tot12

        K=self.my_FEproblem1.mats[-1].props[0]
        G=self.my_FEproblem1.mats[-1].props[1]
    
        tanfi=self.my_FEproblem1.mats[-1].props[10]
        cc=self.my_FEproblem1.mats[-1].props[11]
        tanpsi=self.my_FEproblem1.mats[-1].props[12]
        hcc=self.my_FEproblem1.mats[-1].props[14]
        
        A=-tanfi*K*dot_evol_tot
        
        B=s11*dot_e_tot11+s22*dot_e_tot22+s33*dot_e_tot33+s23*dot_e_tot23+s13*dot_e_tot13+s12*dot_e_tot12
        B=-B.copy()/J2_1.copy()*G
        
        C=tanfi*tanpsi*K-G-cc*hcc
        
        lambda_dot_an=(A+B)/C
    
        # print("21:11",self.my_FEproblem1.array_gp_svars_comp)
        lambda_dot_calc=self.my_FEproblem1.array_gp_svars_comp[:,38]
        #
        # print(lambda_dot_an)
        # print(lambda_dot_calc)
        # values_diff=lambda_dot_an[-2]-lambda_dot_calc[-2]
        
        values_diff=lambda_dot_an[lambda_dot_calc>10**(-8)]-lambda_dot_calc[lambda_dot_calc>10**(-8)]

        
        # print('Alex')
        # print(J2_1+tanfi*p_eff-cc*(1+hcc*eq_p))
        # print(lambda_dot_calc)
        print(values_diff)
  
        equal=abs(np.linalg.norm(values_diff))<=1.e-6
        self.assertTrue(equal , "Not identical ldot: "+str(abs(np.linalg.norm(values_diff))))

####################################################################################3    
    # def test_analytical__yield_stress(self):  
    #     self.my_FEproblem1.extract_force_disp()
    #
    #     print(self.my_FEproblem1.extract_force_disp().size)
    #     print(self.my_FEproblem1.extract_force_disp().shape)
    #
    #
    #     values_time1 = self.my_FEproblem1.array_time
    #     values_disp0 = self.my_FEproblem1.array_disp[:,0]
    #     values_disp1 = self.my_FEproblem1.array_disp[:,1]
    #
    #     self.my_FEproblem1.extract_force_disp()  
    #     self.my_FEproblem1.extract_svars_gauss_point()  
    #
    #     gamma_dot_0=values_disp0/1.
    #     gamma_dot_5=values_disp1/1.
    #     # print(self.my_FEproblem1.array_gp_svars_comp.size)
    #     # print(self.my_FEproblem1.array_gp_svars_comp.shape)
    #     #
    #     # gamma_dot_0=self.my_FEproblem1.array_gp_svars_comp[:,1]
    #     # gamma_dot_5=self.my_FEproblem1.array_gp_svars_comp[:,2]
    #
    #     print(gamma_dot_0)
    #     print(gamma_dot_5)
    #
    #     K=self.my_FEproblem1.mats[-1].props[0]
    #     G=self.my_FEproblem1.mats[-1].props[1]
    #
    #     tanfi=self.my_FEproblem1.mats[-1].props[10]
    #     cc=self.my_FEproblem1.mats[-1].props[11]
    #     tanpsi=self.my_FEproblem1.mats[-1].props[12]
    #     hcc=self.my_FEproblem1.mats[-1].props[14]
    #
    #     dalpha=(G*gamma_dot_5-K*tanfi*gamma_dot_0*(np.sqrt(2)/2))/(G-K*tanfi*tanpsi+cc*hcc)
    #     print(dalpha)
    #     dot_sigma12=2.*G*(gamma_dot_5+dalpha)
    #     print(dot_sigma12)
    #     tau_yield_anal_1=cc+dot_sigma12
    #
    #     values=tau_yield_anal_1
    #     print(values)
    #
    #     with open(reference_data_path+"/Cauchy2D_Drucker_Prager_stress_App_3.out", "wb") as fp:   #Pickling
    #         pickle.dump(values,fp)    
    #
    #     with open(reference_data_path+"/Cauchy2D_Drucker_Prager_stress_App_3.out", "rb") as fp:   #Pickling
    #         values_ref=pickle.load(fp)        
    #
    #     values_diff=values_ref-values
    #     equal=abs(np.linalg.norm(values_diff))<=1.e-8
    #     self.assertTrue(equal, "Not identical_analytical_stress: "+str(abs(np.linalg.norm(values_diff))))  
    #
    # def test_analytical__yield_stress_comparison(self):  
    #     self.my_FEproblem1.extract_force_disp()  
    #     self.my_FEproblem1.extract_svars_gauss_point()  
    #
    #     gamma_dot_2=np.divide(self.my_FEproblem1.array_gp_svars_comp,self.my_FEproblem1.array_dtime)
    #     gamma_dot_5=np.divide(self.my_FEproblem1.array_gp_svars_comp,self.my_FEproblem1.array_dtime)
    #
    #     K=self.my_FEproblem1.mats[-1].props[0]
    #     G=self.my_FEproblem1.mats[-1].props[1]
    #
    #     tanfi=self.my_FEproblem1.mats[-1].props[10]
    #     cc=self.my_FEproblem1.mats[-1].props[11]
    #     tanpsi=self.my_FEproblem1.mats[-1].props[12]
    #     hcc=self.my_FEproblem1.mats[-1].props[14]
    #
    #     dot_sigma12=2*G*(gamma_dot_5-(G*gamma_dot_5)-K*tanfi*gamma_dot_2*(np.sqrt(2)/2))/(G-K*tanfi*tanpsi+cc*hcc)
    #     tau_yield_anal_1=cc+dot_sigma12
    #
    #     values_force1=self.my_FEproblem1.array_force
    #     diff_values1=values_force1[-1]+tau_yield_anal_1[-1]
    #
    #     equal=abs(np.linalg.norm(diff_values1))<=1.e-8
    #     self.assertTrue(equal, "Not identical_analytical_stress_compare_1: "+str(abs(np.linalg.norm(diff_values1))))
         
if __name__ == '__main__':
    unittest.main()