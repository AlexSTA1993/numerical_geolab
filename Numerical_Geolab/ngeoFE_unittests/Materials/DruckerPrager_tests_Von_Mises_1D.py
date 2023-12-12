'''
Created on Sep 11, 2018

@author: Ioannis Stefanou
'''

'''
Created on Sep 11, 2018

@author: Ioannis Stefanou
'''
import numpy as np
from ngeoFE.materials import UserMaterial
from math import sqrt
import unittest
import pickle
import matplotlib.pyplot as plt
from ngeoFE_unittests import ngeo_parameters
ngeo_parameters.reference_data_path='/home/alexandrosstathas/eclipse-workspace/numerical_geolab/Numerical_Geolab/ngeoFE_unittests/Materials/reference_data/'


import os

print(os.getcwd())

def set_material_1_properties(EE,nu,cc,tanfi,tanpsi,Hsfi,Hscc):
    """
    Sets material parameters
    """
    GG=EE/(2.*(1.+nu))
    KK=EE*GG/(3.*(3.*GG-EE))
    props=np.array([KK,GG,0.,0.,0.,0.,0.,0.,0.,0.,tanfi,cc,tanpsi,Hsfi,Hscc,0.,0.,0.,0000.])
    props=props.astype("double")
    return props

def p(stress):
    p=stress[0]+stress[1]+stress[2]
    return p/3.

def q(stress):
    q=stress[0]**2+stress[1]**2+stress[2]**2
    q+=3.*(stress[3]**2+stress[4]**2+stress[5]**2)
    q-=stress[0]*stress[1]+stress[1]*stress[2]+stress[2]*stress[0]
    return sqrt(abs(q)/3.)

def ev(deformation):
    ev=deformation[0]+deformation[1]+deformation[2]
    return ev

def eq(deformation):
    eq=4.*(deformation[0]**2+deformation[1]**2+deformation[2]**2)
    eq+=12.*(deformation[3]**2+deformation[4]**2+deformation[5]**2)
    eq-=4.*(deformation[0]*deformation[1]+deformation[1]*deformation[2]+deformation[2]*deformation[0])
    return sqrt(abs(eq)/3.)

def lambdaanal(gammadot,G,cc,hcc):
    return 4.*G*gammadot/(4.*G+cc*hcc)#2.*G*gammadot/(G+cc*hcc)

class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''
        Run a load path for Drucker Prager Material
        '''
        print('unittest_started')
        cls.notfirsttime=True
        env_lib=ngeo_parameters.env_lib        #umat_lib='./libplast_Cauchy3D-DP.so'
        umat_lib_path= ngeo_parameters.umat_lib_path
        umat_lib = umat_lib_path+'CAUCHY3D-DP/libplast_Cauchy3D-DP.so'
        umat_id=1       # if many materials exist in the same library
        mat=UserMaterial(env_lib,umat_lib,umat_id)
        mat.props=set_material_1_properties(20.e3,0.,80.,0.,0.,0.,-5.)

        increments=100
        e11min=0.;e11max=0.0;deps11=(e11max-e11min)/increments
        e22min=0.;e22max=0.0;deps22=(e22max-e22min)/increments
        e12min=0.;e12max=0.1;deps12=(e12max-e12min)/increments
        
        deps=[np.array([0.,0.])]
        for i in range(increments):
            deps.append(np.array([deps11,deps12]))
        #print(deps)
        
        stressGP_t=np.zeros(2)
        svarsGP_t=np.zeros(29)
        dsdeGP_t=np.zeros(2**2)
        dt=1.
        
        stress=[];ps=[];qs=[]
        epsilon=[];evs=[];eqs=[];lambdadot=[];lambdaan=[]
        for i in range(len(deps)):
            #print("Increment: ",i)
            deGP=deps[i][:].copy()
            nill=mat.usermatGP(stressGP_t,deGP, svarsGP_t, dsdeGP_t, dt,0)
            if nill!=1:
                svarsGP_t1=svarsGP_t.copy()
                ps.append(p(svarsGP_t1[0:6]))
                qs.append(q(svarsGP_t1[0:6]))
                stress.append(svarsGP_t1[0:6])
                evs.append(ev(svarsGP_t1[6:12]))
                eqs.append(eq(svarsGP_t1[6:12]))
                epsilon.append(svarsGP_t1[6:12])
                lambdadot.append(svarsGP_t1[21])
                lambdaan.append(lambdaanal(svarsGP_t1[27],10.e3,80.,-5.))
            else:
                print("material problem")
                return


         
        plt.plot(evs, ps, "bo-", label="$p-\epsilon_v$")
        plt.ylabel("$p$")
        plt.xlabel("$\epsilon_v$")
        plt.legend()
        plt.show()
         
        plt.plot(eqs, qs, "bo-", label="$q-\epsilon_q$")
        plt.ylabel("$q$")
        plt.ylim(ymax=1.e2)
        plt.xlabel("$\epsilon_q$")
        plt.legend()
        plt.show()
        
        cls.stress=stress
        cls.ps=ps
        cls.qs=qs
        cls.epsilon=epsilon
        cls.evs=evs
        cls.eqs=eqs
        cls.lambdadot=lambdadot
        cls.lambdaan=lambdaan
        cls.deps=deps
        
    def test_stresses(self):
        '''
        Tests Drucker Prager material (stresses)
        '''
        reference_data_path = ngeo_parameters.reference_data_path  
        reference_data = reference_data_path+'DP_1D_stress_values.out'
        values= np.array(self.stress)
        with open(reference_data, "wb") as fp:   #Pickling
            pickle.dump(values, fp)
 
          
        with open(reference_data, "rb") as fp:   # Unpickling
            values_ref = pickle.load(fp)
        
        values_diff=values_ref-values
        
        print(abs(np.linalg.norm(values_diff)))
        
        equal=abs(np.linalg.norm(values_diff))<=1.e-10
        
        self.assertTrue(equal, "Not identical stresses: "+str(abs(np.linalg.norm(values_diff))))
        
    def test_total_deformations(self):
        '''
        Tests Drucker Prager material (deformations)
        '''
        reference_data_path = ngeo_parameters.reference_data_path  
        reference_data = reference_data_path+'DP_1D_epsilon_values.out'
        values= np.array(self.epsilon)
        with open(reference_data, "wb") as fp:   #Pickling
            pickle.dump(values, fp)
        
        reference_data_path = ngeo_parameters.reference_data_path 
        print(reference_data_path)   
       
        with open(reference_data, "rb") as fp:   # Unpickling
            values_ref = pickle.load(fp)
            
        values_diff=values_ref-values
        equal=abs(np.linalg.norm(values_diff))<=1.e-13
        
        self.assertTrue(equal, "Not identical deformations: "+str(abs(np.linalg.norm(values_diff))))
    
    def test_dot_lambda(self):
        '''
        Tests Von_Mises material (lambda)
        '''
        print(self.lambdadot)
        print(self.lambdaan)
        
        values_diff=np.array(self.lambdadot)-np.array(self.lambdaan)
        
        equal=abs(np.linalg.norm(values_diff))<=1.e-13
        
        self.assertTrue(equal, "Not identical lambda: "+str(abs(np.linalg.norm(values_diff))))
        
        
    def test_Gep(self):
        '''
        Tests Von_Mises material (elasto-plastic matrix)
        '''
        G=10.e3
        cc=80.
        hcc=-5.
        
        dstress_final=self.stress[-1][-1]-self.stress[-2][-1]
        depsilon_final=self.deps[-1][-1]
        
        # print(self.stress)
        # print(self.deps)
        
        Gep_num=(dstress_final)/(depsilon_final)
        Gep_an = G*cc*hcc/(G+cc*hcc)
        
        print('hello')
        # print(Gep_num)
        # print(Gep_an)
        
        values_diff=Gep_num-Gep_an
        
        equal=abs(np.linalg.norm(values_diff))<=1.e-13
        
        self.assertTrue(equal, "Not identical Gep: "+str(abs(np.linalg.norm(values_diff))))
        
        
if __name__ == '__main__':
    unittest.main()