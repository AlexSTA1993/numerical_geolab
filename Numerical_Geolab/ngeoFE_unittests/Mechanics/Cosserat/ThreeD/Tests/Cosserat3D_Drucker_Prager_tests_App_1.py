'''
Created on Nov 5, 2018

@author: Alexandros Stathas

Contains unit tests of ngeoFE applied to Cauchy continua in 1D linear elasticity

'''
import os
import unittest
from ngeoFE_unittests.Mechanics.Cosserat.ThreeD.BVP.Cosserat3D_Drucker_Prager_App_1 import Cosserat3DFEformulation, Cosserat3DFEproblem

from dolfin import *

from dolfin.cpp.io import HDF5File

import pickle
import numpy as np

from ngeoFE_unittests import ngeo_parameters
from ngeoFE_unittests import plotting_params 

reference_data_path = ngeo_parameters.reference_data_path    

def calculate_deviatoric_tensor(sigma11,sigma12,sigma13,sigma21,sigma22,sigma23,sigma31,sigma32,sigma33):
    p_tot=(sigma11+sigma22+sigma33)/3.
    s11=sigma11-p_tot#+PF
    s12=sigma12
    s13=sigma13
    
    s21=sigma21
    s22=sigma22-p_tot#+PF
    s23=sigma23
    
    s31=sigma31
    s32=sigma32
    s33=sigma33-p_tot#+PF
        
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

# def product_sij_sji(sij,sji):
#     # J2=sij[:,0,0]*sij[:,0,0]+sij[:,1,1]*sij[:,1,1]+sij[:,2,2]*sij[:,2,2]
#     J2=np.zeros((sij.shape[0]))
#     for k in range(sij.shape[0]):
#         sum=0.
#         for i in range(3):
#             for j in range(3):
#                 sum+=sij[k,i,j]*sji[k,i,j]
#         J2[k]=sum
#     return J2
    
def calculate_J2(h1,h2,sigma11,sigma12,sigma13,sigma21,sigma22,sigma23,sigma31,sigma32,sigma33):
    sij=calculate_deviatoric_tensor(sigma11,sigma12,sigma13,sigma21,sigma22,sigma23,sigma31,sigma32,sigma33)
    
    sji=np.array(np.zeros(sij.shape))
    sji[:]=np.transpose(sij,axes=(0,2,1))
    
    J2_1=h1*tensor_product(sij,sij)
    
    J2_2=h2*tensor_product(sij,sji)
    return J2_1+J2_2

def calculate_eq_tot(g1,g2,epsilon11,epsilon12,epsilon13,epsilon21,epsilon22,epsilon23,epsilon31,epsilon32,epsilon33):
    eij=calculate_deviatoric_tensor(epsilon11,epsilon12,epsilon13,epsilon21,epsilon22,epsilon23,epsilon31,epsilon32,epsilon33)
    
    eji=np.array(np.zeros(eij.shape))
    eji[:]=np.transpose(eij,axes=(0,2,1))
    
    eq_tot_1=g1*tensor_product(eij,eji)
    eq_tot_2=g2*tensor_product(eij,eji)

    return eq_tot_1+eq_tot_2

def assign_generalized_stress(svars_values,start=0,step=6,total_comp=9):
    # print(start,step,total_comp)
    # print(svars_values.shape)
    g_svars=[]
    for i in range(start,start+total_comp):
        # print(i)
        # print(g_svars)
        g_svars.append(svars_values[:,int(i*step)])
        
    # print(g_svars)
    # print(len(g_svars))
    return g_svars #g_svars[0], g_svars[1], g_svars[2], g_svars[3], g_svars[4], g_svars[5], g_svars[6], g_svars[7], g_svars[8] 

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
    
    # print(sji[-1])
    
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
        cls.my_FEformulation=Cosserat3DFEformulation()
        cls.my_FEproblem1=Cosserat3DFEproblem(cls.my_FEformulation)
        cls.my_FEproblem1.give_me_solver_params(scale_t=1.)
        cls.converged1=cls.my_FEproblem1.run_analysis_procedure(reference_data_path)       
        
    def test_execution(self):
        '''
        Tests execution and convergence
        '''
        self.assertTrue(self.converged1, "Convergence failed")
    
    def test_shear_force_displacement_values(self):
        '''
        Tests calculated nodal forces and displacements to values in ./reference_data/Cosserat3D_Drucker_Prager_force_disp_values.out
        '''
        self.my_FEproblem1.extract_force_disp()
        values_time1 = self.my_FEproblem1.array_time
        values_force1 = self.my_FEproblem1.array_force
        values_disp1 = self.my_FEproblem1.array_disp

        values1=np.concatenate((values_time1, values_disp1, values_force1), axis=1)
 
        with open(reference_data_path+"Cosserat3D_Drucker_Prager_force_disp_values_App_1.out", "wb") as fp:   #Pickling
            pickle.dump(values1,fp)        
        

        with open(reference_data_path+"Cosserat3D_Drucker_Prager_force_disp_values_App_1.out", "rb") as fp:   #Pickling
            values_ref=pickle.load(fp)        
        values_diff_1=values_ref-values1
 
        self.my_FEproblem1.extract_svars_gauss_point()
        svars_values=self.my_FEproblem1.array_gp_svars_comp
        
        with open(reference_data_path+"Cosserat3D_Drucker_Prager_svars_values_App_1.out", "wb") as fp:   #Pickling
            pickle.dump(svars_values,fp)        
    
        with open(reference_data_path+"Cosserat3D_Drucker_Prager_svars_values_App_1.out", "rb") as fp:   #Pickling
            svars_values_ref=pickle.load(fp)        
        values_diff_2=svars_values_ref-svars_values
  
    
        equal_1=abs(np.linalg.norm(values_diff_1))<=1.e-13
        equal_2=abs(np.linalg.norm(values_diff_2))<=1.e-13
        self.assertTrue(equal_1 and equal_2, "Not identical time, displacements, forces, svars: "+str(abs(np.linalg.norm(values_diff_1)))+str(abs(np.linalg.norm(values_diff_2))))
        #asserts that data are correct
        #if dtat are correct then plot diagram
        if equal_1 and equal_2:
            nz=self.my_FEproblem1.nz
            K, G, Gc, M, Mc, R,tanfi,cc,tanpsi,Hsfi,Hscc,h1,h2,h3,h4=assign_material_parameters(self.my_FEproblem1.mats[0])

            sigma=assign_generalized_stress(svars_values,start=0,step=6,total_comp=9)
            sij= calculate_deviatoric_tensor(*sigma)

            mu=assign_generalized_stress(svars_values,start=9,step=6,total_comp=9)
            mij= calculate_deviatoric_tensor(*mu)

            epsilon=assign_generalized_stress(svars_values,start=18,step=6,total_comp=9)
            kappa=assign_generalized_stress(svars_values,start=27,step=6,total_comp=9)
            
            g1=self.my_FEproblem1.mats[0].props[19]
            g2=self.my_FEproblem1.mats[0].props[20]
            g3=self.my_FEproblem1.mats[0].props[21]
            g4=self.my_FEproblem1.mats[0].props[22]
            

            J2_1=calculate_J2(h1,h2,*sigma)
            J2_2=calculate_J2(h3,h4,*mu)
            
            J2=np.sqrt(J2_1+1/R**2*J2_2)

            eq_tot_1=calculate_eq_tot(g1,g2,*epsilon)
            eq_tot_2=calculate_eq_tot(g3,g4,*kappa)
            eq_tot=np.sqrt(eq_tot_1+R**2*eq_tot_2)
            
            p_eff=(sigma[0]+sigma[4]+sigma[8])/3.
            # print(p_eff)
            evol_tot_1=epsilon[0]+epsilon[4]+epsilon[8]           
            
            x1=list(values_time1[:].copy())
            y1=list(J2.copy())
            x1.insert(0,0)
            y1.insert(0,0.)
            
            x2=list(values_time1[:].copy())
            y2=list(evol_tot_1.copy())
            x2.insert(0,0)
            y2.insert(0,0.)
            
            x3=list(p_eff.copy())
            x3.insert(0,p_eff[0])
    
            fig, ax1, color1 = plotting_params.object_plot_axes('$t$ [s]', y1_txt='$\sqrt{J_2}$ [MPa]',color1='k',y2_txt='',color2='', title='',mode='1')
            plotting_params.object_plot(x1, y1,'y2', ax1, 'ax2','x2', mode='1',color1=color1,color2='',label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cosserat_3D_Drucker-Prager_App_1_J2_t',mode='1')
            
            x1=list(eq_tot.copy())
            y1=list(J2.copy())
            x1.insert(0,0)
            y1.insert(0,0)
        
            fig, ax1, color1 = plotting_params.object_plot_axes('$q^{tot}$', y1_txt='$\sqrt{J_2}$ [MPa]',color1='k',y2_txt='',color2='', title='',mode='1')
            plotting_params.object_plot(x1, y1,'y2', ax1, 'ax2','x2', mode='1',color1='r',color2='b',label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cosserat_3D_Drucker-Prager_App_1_J2_q',mode='1')
            
            fig, ax1, color1 = plotting_params.object_plot_axes('$p$ [MPa]', y1_txt='$\sqrt{J_2}$ [MPa]',color1='k',y2_txt='',color2='', title='',mode='1')
            plotting_params.object_plot(x3, y1,'y2', ax1, 'ax2','x2', mode='1',color1='r',color2='b',label_string='')
            plotting_params.show_plot()
            plotting_params.plot_legends(ngeo_parameters.reference_data_path, fig, filename='Cosserat_3D_Drucker-Prager_App_1_J2_p',mode='1')
    
    def test_analytical__lambda_dot(self):  
        
        self.my_FEproblem1.extract_svars_gauss_point()
        svars_dtime=self.my_FEproblem1.array_dtime
        svars_values=self.my_FEproblem1.array_gp_svars_comp
        # print(svars_values.shape)
        
        K, G, Gc, M, Mc, R,tanfi,cc,tanpsi,Hsfi,Hscc,h1,h2,h3,h4=assign_material_parameters(self.my_FEproblem1.mats[0])

        # sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33=assign_generalized_stress(svars_values,start=0,step=2,total_comp=9)
        # sij= calculate_deviatoric_tensor(sigma11,sigma12,sigma13,sigma21,sigma22,sigma23,sigma31,sigma32,sigma33)
        sigma=assign_generalized_stress(svars_values,start=0,step=6,total_comp=9)
        sij=calculate_deviatoric_tensor(*sigma)
        # print(sij[-1])
        # mu11, mu12, mu13, mu21, mu22, mu23, mu31, mu32, mu33=assign_generalisezed_stress(svars_values,start=9,step=2,total_comp=9)
        # mij= calculate_deviatoric_tensor(mu11, mu12, mu13, mu21, mu22, mu23, mu31, mu32, mu33)

        mu=assign_generalized_stress(svars_values,start=9,step=6,total_comp=9)
        mij= calculate_deviatoric_tensor(*mu)
        # print(mij[-1])
        
        J2_1=calculate_J2(h1,h2,*sigma)    
        J2_2=calculate_J2(h3,h4,*mu)
            
        J2=np.sqrt(J2_1+1/R**2*J2_2)
        
        # print(J2)
        
        p_eff=(sigma[0]+sigma[4]+sigma[8])/3.
 
        tau=J2#+tanfi*p_eff-cc
        tauRR=tau*R**2
        
        bkl_F= betakl(K,tanfi,G,Gc,h1,h2,tau,sij)
        # print(bkl_F[-2])
        
        bkl_FM= betakl(K,0,M,Mc,h3,h4,tauRR,mij)
        
        Hp_F=calculate_Hp(K,G,Gc,tanfi,tanpsi,sij,tau,h1,h2)
        Hp_FM=calculate_Hp(K,M,Mc,0,0,mij,tauRR,h3,h4)
        
        # print(Hp_F)
        Hp=Hp_F+Hp_FM
        
        # print(svars_dtime.reshape((-1,)))
        dot_u1=self.my_FEproblem1.u1_tot*svars_dtime.reshape((-1,))#self.my_FEproblem1.slv.dtmax
        dot_gkl=apply_total_gen_strain_rate(svars_dtime,0,0,0,0,0,dot_u1,0,0,0)
        dot_kkl=apply_total_gen_strain_rate(svars_dtime,0,0,0,0,0,0,0,0,0)
        
        # print(dot_gkl[-2])
    
        # print(dot_kkl)
        
        # print(product_sij_sji(bkl_F,dot_gkl))
        
        lambda_dot_an=1./Hp*(tensor_product(bkl_F,dot_gkl)+tensor_product(bkl_FM,dot_kkl))
    
        FF=J2+tanfi*p_eff

        lambda_dot_an=np.where((FF>0),lambda_dot_an,0.)
        # print(lambda_dot_an)
        lambda_dot_calc=self.my_FEproblem1.array_gp_svars_comp[:,-1]

        print(lambda_dot_an)
        print(lambda_dot_calc)
        values_diff=lambda_dot_an[lambda_dot_calc>10**(-8)]-lambda_dot_calc[lambda_dot_calc>10**(-8)]
    
        print(values_diff)
    
        equal=abs(np.linalg.norm(values_diff))<=1.e-4
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