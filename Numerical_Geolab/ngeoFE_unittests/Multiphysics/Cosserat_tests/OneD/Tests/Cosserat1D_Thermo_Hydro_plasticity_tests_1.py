'''
Created on Mai 30, 2022

@author: Alexandros Stathas

Contains unit tests of ngeoFE applied to Cosserat continua in 1D Thermo-Hydroplasticity with a 
Drucker Prager yield criterion. Check softening behavior due to pore fluid pressure increase due to thermal
pressurization. The line segment is under 1D uniaxial compression.

Checks:
-Convergence
-Generalised force displacement values
-Analytical vs numerical total stress
-Final temperature and pore fluid pressure values
-Rate of temperature and pore fluid pressure increase'''
import sys
import os
import unittest
from ngeoFE_unittests.Multiphysics.Cosserat_tests.OneD.BVP.Cosserat1D_Drucker_Prager_Thermo_Hydro_Plasticity_1 import CosseratTHM1DFEformulation,CosseratTHM1DFEproblem

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
    return [g_svars[0], g_svars[1], g_svars[2], g_svars[3], g_svars[4], g_svars[5], g_svars[6], g_svars[7], g_svars[8]] 

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

def dissipation(sigma,mu,dot_e_p,dot_k_p):
    sigma=np.array(sigma)
    mu=np.array(mu)
    dot_e_p=np.array(dot_e_p)
    dot_k_p=np.array(dot_k_p)
        
    W1=np.sum(sigma[:,0:]*dot_e_p[:,0:],axis=0)
    
    W2=np.sum(mu[:,0:]*dot_k_p[:,0:],axis=0)
    
    W=W1+W2
    return W

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
    
        # with open(reference_data_path+"Cosserat1D_THM_Thermo_Hydro_plasticity_uniaxial_comp.out", "wb") as fp:   #Pickling
        #     pickle.dump(values1,fp)        
    
        with open(reference_data_path+"Cosserat1D_THM_Thermo_Hydro_plasticity_uniaxial_comp.out", "rb") as fp:   #Pickling
            values_ref=pickle.load(fp)        
        values_diff=values_ref-values1
    
        self.my_FEproblem1.extract_svars_gauss_point()
        svars_values=self.my_FEproblem1.array_gp_svars_comp
       
    
        equal=abs(np.linalg.norm(values_diff))<=1.e-9 #docker precision is reduced it however better than the accuracy of the equlibrium solver 10e-6 and material solver 10e-8
        self.assertTrue(equal, "Not identical time, displacements, forces: "+str(abs(np.linalg.norm(values_diff))))
        #asserts that data are correct
        #if data are correct then plot diagram
        if equal and activate_plots:
            
            K, G, Gc, M, Mc, R,tanfi,cc,tanpsi,Hsfi,Hscc,h1,h2,h3,h4=assign_material_parameters(self.my_FEproblem1.mats[0])

            [sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33]=assign_generalisezed_stress(svars_values,start=0,step=1,total_comp=9)
            sij= calculate_deviatoric_tensor(sigma11,sigma12,sigma13,sigma21,sigma22,sigma23,sigma31,sigma32,sigma33)

            [mu11, mu12, mu13, mu21, mu22, mu23, mu31, mu32, mu33]=assign_generalisezed_stress(svars_values,start=9,step=1,total_comp=9)
            mij= calculate_deviatoric_tensor(mu11, mu12, mu13, mu21, mu22, mu23, mu31, mu32, mu33)

            [epsilon11, epsilon12, epsilon13, epsilon21, epsilon22, epsilon23, epsilon31, epsilon32, epsilon33]=assign_generalisezed_stress(svars_values,start=24,step=1,total_comp=9)
            [kappa11, kappa12, kappa13, kappa21, kappa22, kappa23, kappa31, kappa32, kappa33]=assign_generalisezed_stress(svars_values,start=33,step=1,total_comp=9)
            
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
            
            
            x1=list(values_time1[:].copy())
            y1=list(values_gen_disp1[:,-1].copy())
            x1.insert(0,0)
            y1.insert(0,0)

            x2=list(values_time1[:].copy())
            y2=list(values_gen_disp1[:,-2].copy())
            x2.insert(0,0)
            y2.insert(0,self.my_FEproblem1.Pressure_loading)

            fig, ax1, ax2, color1, color2 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$T\; $[$^o$  C]',color1='',y2_txt='$P$ [MPa]',color2='', title='',mode='2')
            
            plotting_params.object_plot(x1, y1, y2, ax1, ax2,x1, mode='3',color1=color1,color2=color2, label_string='')

            plotting_params.show_plot()           
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cosserat_1D_Thermo_Hydro_plasticity_uniaxial_comp_T_P',mode='1')

    
            x1=list(values_time1[:].copy())
            y1=list(values_gen_disp1[:,0].copy()*self.my_FEproblem1.scale_u)
            x1.insert(0,0)
            y1.insert(0,0)

            fig, ax1, color1 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$u_z$ [mm]',color1='',y2_txt='',color2='', title='',mode='1')
            plotting_params.object_plot(x1, y1, "y1", ax1, 'ax2', mode='1',color1=color1,color2='color2',label_string='')
            plotting_params.show_plot()           
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cosserat_1D_Thermo_Hydro_plasticity_uniaxial_comp_u',mode='1')

            x2=list(values_time1[:].copy())
            y2=list(J2_1_f)
            x2.insert(0,0)
            y2.insert(0,0)

            x3=list(values_time1[:].copy())
            y3=list(p_eff)
            x3.insert(0,0)
            y3.insert(0,self.my_FEproblem1.Normal_loading_eff)

            fig, ax1, ax2, color1, color2 = plotting_params.object_plot_axes('$t$ [s]',y1_txt= '$J_2$ [MPa]',color1='',y2_txt='$P^\prime$ [MPa]',color2='', title='',mode='2')
            
            plotting_params.object_plot(x1, y2, y3, ax1, ax2, mode='2',color1=color1,color2=color2, label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cosserat_1D_Thermo_Hydro_plasticity_uniaxial_comp_J_2_P_eff',mode='1')
           
    
    def test_final_stress(self):
        K=self.my_FEproblem1.mats[0].props[0]
        G=self.my_FEproblem1.mats[0].props[1]
    
        self.my_FEproblem1.extract_svars_gauss_point()
        svars=self.my_FEproblem1.array_gp_svars_comp
        self.sigma11=svars[:,0]
        self.sigma22=svars[:,4]
        self.sigma33=svars[:,8]
        self.epsilon11=svars[:,24]
        self.epsilon22=svars[:,28]
        self.epsilon33=svars[:,32]
        self.epsilon11_pl=svars[:,66]
        self.epsilon22_pl=svars[:,67]
        self.epsilon33_pl=svars[:,68]
        a=4*G/3+K
        b=-2*G/3+K
        
        self.epsilon11_e=self.epsilon11-self.epsilon11_pl
        self.epsilon22_e=self.epsilon22-self.epsilon22_pl
        self.epsilon33_e=self.epsilon33-self.epsilon33_pl
        
        self.dsigma_calc=(a*self.epsilon11_e+b*self.epsilon22_e+b*self.epsilon33_e)
        self.sigma_calc=self.my_FEproblem1.Normal_loading_eff+ self.dsigma_calc
        
        self.my_FEproblem1.extract_generalized_force_disp()
        values_gen_disp1 = self.my_FEproblem1.array_gen_disp
        self.P=values_gen_disp1[:,-2]

       
        self.sigma_tot=self.sigma_calc-self.P
        self.sigma_tot_num= self.sigma11-self.P
        
        values_diff=self.sigma_tot.copy()-self.sigma_tot_num.copy()
        equal=abs(np.linalg.norm(values_diff))<=1.e-3
        self.assertTrue(equal, "Not identical total stress an vs num: "+str(np.linalg.norm(values_diff)))    

    def test_final_T_P(self):
        self.my_FEproblem1.extract_generalized_force_disp()
        values_gen_disp1 = self.my_FEproblem1.array_gen_disp
        self.P=values_gen_disp1[:,-2]
        self.T=values_gen_disp1[:,-1]
        values_diff=self.P-self.T-self.my_FEproblem1.Pressure_loading+values_gen_disp1[:,0]*self.my_FEproblem1.scale_u

        equal=abs(sum(values_diff))<=1.e-6
        self.assertTrue(equal, "Not identical total stress an vs num: "+str(sum(values_diff)))    

    
    def test_DT_dissipation_equality(self):

        self.my_FEproblem1.extract_svars_gauss_point()
        svars_values=self.my_FEproblem1.array_gp_svars_comp
        
        sigma=assign_generalisezed_stress(svars_values,start=0,step=1,total_comp=9)

        mu=assign_generalisezed_stress(svars_values,start=9,step=1,total_comp=9)


        dot_e_p=assign_generalisezed_stress(svars_values,start=48,step=1,total_comp=9)
        dot_k_p=assign_generalisezed_stress(svars_values,start=57,step=1,total_comp=9)

        dissip1= dissipation(sigma,mu,dot_e_p,dot_k_p)

        rhoC=self.my_FEproblem1.mats[0].props[11]
        dT_calc=1./rhoC*dissip1
        
        dT=self.my_FEproblem1.array_gp_svars_comp[:,-1]

        values_diff=dT[1:]-dT_calc[:-1]
        
        equal=abs(np.linalg.norm(values_diff))<=1.e-6
        self.assertTrue(equal, "Not identical temperature increment: "+str(abs(np.linalg.norm(values_diff))))    

    # def test_DT_DP_equality(self):
    #     self.my_FEproblem1.extract_generalized_force_disp()
    #     values_time1 = self.my_FEproblem1.array_time
    #     values_gen_disp1 = self.my_FEproblem1.array_gen_disp
    #
    #     tanfi=self.my_FEproblem1.mats[0].props[15]
    #     R=self.my_FEproblem1.mats[0].props[6]
    #     h1=self.my_FEproblem1.mats[0].props[20]
    #     h2=self.my_FEproblem1.mats[0].props[21]
    #     h3=self.my_FEproblem1.mats[0].props[22]
    #     h4=self.my_FEproblem1.mats[0].props[23]    
    #
    #     self.my_FEproblem1.extract_svars_gauss_point()
    #     svars=self.my_FEproblem1.array_gp_svars_comp
    #
    #     [sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33]=assign_generalisezed_stress(svars,start=0,step=1,total_comp=9)
    #
    #     [mu11, mu12, mu13, mu21, mu22, mu23, mu31, mu32, mu33]=assign_generalisezed_stress(svars,start=9,step=1,total_comp=9)
    #
    #     J2_1=calculate_J2(sigma11,sigma12,sigma13,sigma21,sigma22,sigma23,sigma31,sigma32,sigma33,h1,h2)
    #     J2_2=calculate_J2(mu11,mu12,mu13,mu21,mu22,mu23,mu31,mu32,mu33,h3,h4)
    #
    #     J2_1_f=np.sqrt(J2_1+1/R**2*J2_2)
    #     p_eff=(sigma11+sigma22+sigma33)/3.
    #
    #     F=J2_1_f-tanfi*p_eff
    #
    #     self.dP=svars[:,-2]
    #     self.dT=svars[:,-1]
    #
    #     self.evol_dt=values_gen_disp1[:,0]*self.my_FEproblem1.scale_u*self.my_FEproblem1.slv.dtmax #10 number of increments
    #
    #     lstar=self.my_FEproblem1.mats[0].props[13]
    #     bstar=self.my_FEproblem1.mats[0].props[9]
    #
    #     values_diff=self.dP[:]-lstar/bstar*self.dT[:]+self.evol_dt[:]/bstar
    #     values_diff=np.where(F[:]<0,0.,values_diff)
    #
    #     equal=abs(sum(values_diff))<=1.e-8
    #     self.assertTrue(equal, "Not identical temperature increment: "+str(sum(values_diff)))    
    #
    #     if equal and activate_plots:
    #         x1=list(values_time1[:].copy())
    #         y1=list(self.dT)
    #         x1.insert(0,0)
    #         y1.insert(0,0)
    #
    #         x2=list(values_time1[:].copy())
    #         y2=list(self.dP)
    #         x2.insert(0,0)
    #         y2.insert(0,0)
    #
    #         fig, ax1, ax2, color1, color2 = plotting_params.object_plot_axes('$t$ [s]',y1_txt= '$\Delta T\; $[$^o$  C]',color1='',y2_txt='$\Delta P$ [MPa]',color2='', title='',mode='2')
    #
    #         plotting_params.object_plot(x1, y1, y2, ax1, ax2,x1, mode='3',color1=color1,color2=color2, label_string='')
    #         plotting_params.show_plot()
    #         plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cosserat_1_Thermo_Hydro_plasticity_uniaxial_comp_DT_DP',mode='1')
    #
    #         plotting_params.show_plot()


if __name__ == '__main__':
    unittest.main()
