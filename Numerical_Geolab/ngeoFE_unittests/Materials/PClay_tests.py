'''
Created on Sep 11, 2018

@author: Ioannis Stefanou
'''
# from ngeoFE_unittests.Multiphysics.1D_HydroElasticity_test import reference_data
from ngeoFE_unittests import ngeo_parameters
ngeo_parameters.reference_data_path='/home/alexandrosstathas/eclipse-workspace/numerical_geolab/Numerical_Geolab/ngeoFE_unittests/Materials/reference_data/'

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

def set_material_1_properties(EE,nu,cc,tanfi,tanpsi,Hsfi,Hscc):
    """
    Sets material parameters
    """
    GG=EE/(2.*(1.+nu))
    KK=EE*GG/(3.*(3.*GG-EE))
    props=np.array([KK,GG,0.,0.,0.,0.,0.,0.,0.,0.,tanfi,cc,tanpsi,Hsfi,Hscc,0.,0.,0.,0.,0.])
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
    eq+=3.*(deformation[3]**2+deformation[4]**2+deformation[5]**2)
    eq-=4.*(deformation[0]*deformation[1]+deformation[1]*deformation[2]+deformation[2]*deformation[0])
    return sqrt(abs(eq)/3.)

class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''
        Run a load path for Drucker Prager Material
        '''
        cls.notfirsttime=True
        reference_data=ngeo_parameters.umat_lib_path
        env_lib=ngeo_parameters.env_lib#['/usr/lib/x86_64-linux-gnu/liblapack.so']
        umat_lib=reference_data+'/CAUCHY3D-DP/libplast_Cauchy3D-DP.so'#'/mnt/f/DEVELOPMENT/UMATERIALS/CAUCHY3D-DP/libplast_Cauchy3D-DP.so'
        umat_id=2       # if many materials exist in the same library
        mat=UserMaterial(env_lib,umat_lib,umat_id)
        mat.props=set_material_1_properties(20.e3,0.,80.,0.,0.,0.,-5.)
        #print(mat.props)
        increments=100
        e11min=0.;e11max=.1;deps11=(e11max-e11min)/increments
        e22min=0.;e22max=.1;deps22=(e22max-e22min)/increments
        e12min=0.;e12max=0.;deps12=(e12max-e12min)/increments
        
        deps=[np.array([0.,0.,0.])]
        for i in range(increments):
            deps.append(np.array([deps11,deps22,deps12]))
        #print(deps)
        
        stressGP_t=np.zeros(3)
        svarsGP_t=np.zeros(29)
        dsdeGP_t=np.zeros(3**2)
        dt=1.
        
        stress=[];ps=[];qs=[]
        epsilon=[];evs=[];eqs=[]
        for i in range(len(deps)):
            #print("Increment: ",i)
            deGP=deps[i][:].copy()
            nill=mat.usermatGP(stressGP_t,deGP, svarsGP_t, dsdeGP_t, dt,0)
            if nill!=1:
                ps.append(p(svarsGP_t[0:6]))
                qs.append(q(svarsGP_t[0:6]))
                stress.append(svarsGP_t[0:6])
                evs.append(ev(svarsGP_t[6:12]))
                eqs.append(eq(svarsGP_t[6:12]))
                epsilon.append(svarsGP_t[6:12])
            else:
                print("material problem")
                return

        #plt.rc('text', usetex=True)
        #plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
        #plt.ion() 
        plt.plot(ps, qs, "bo-", label="$p-q$")
        plt.xlabel("$p$")
        plt.ylabel("$q$")
        plt.ylim(ymax=1.e2)
        plt.legend()
        plt.show()
         
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
        
        
    def test_stresses(self):
        '''
        Tests PClay material (stresses)
        '''
        values= np.array(self.stress)
#         with open("DP_stress_values.out", "wb") as fp:   #Pickling
#             pickle.dump(values, fp)
        reference_data_path = ngeo_parameters.reference_data_path    
        reference_data = reference_data_path+"DP_stress_values.out"            
        with open(reference_data, "rb") as fp:   # Unpickling
            values_ref = pickle.load(fp)
            
        values_diff=values_ref-values
        
#         equal=abs(np.linalg.norm(values_diff))<=1.e-13
        equal=abs(np.linalg.norm(values_diff))<=1.e-10
        self.assertTrue(equal, "Not identical stresses: "+str(abs(np.linalg.norm(values_diff))))
        
    def test_total_deformations(self):
        '''
        Tests Drucker Prager material (deformations)
        '''
        values= np.array(self.epsilon)
#         with open("DP_epsilon_values.out", "wb") as fp:   #Pickling
#             pickle.dump(values, fp)
        reference_data_path = ngeo_parameters.reference_data_path    
        reference_data = reference_data_path+"DP_epsilon_values.out"                      
        with open(reference_data, "rb") as fp:   # Unpickling
            values_ref = pickle.load(fp)
            
        values_diff=values_ref-values
        
        equal=abs(np.linalg.norm(values_diff))<=1.e-13
        
        self.assertTrue(equal, "Not identical deformations: "+str(abs(np.linalg.norm(values_diff))))
    

        
if __name__ == '__main__':
    unittest.main()