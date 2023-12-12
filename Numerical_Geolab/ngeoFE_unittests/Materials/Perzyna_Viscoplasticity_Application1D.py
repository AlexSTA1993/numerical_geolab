'''
Created on Jan 8, 2022

@author: alexandrosstathas
'''
import numpy as np
from ngeoFE.materials import UserMaterial
from math import sqrt
import unittest
import pickle
import matplotlib.pyplot as plt
from ngeoFE_unittests import ngeo_parameters
ngeo_parameters.reference_data_path='/home/alexandrosstathas/eclipse-workspace/numerical_geolab/Numerical_Geolab/ngeoFE_unittests/Materials/reference_data/'


def set_material_1_properties():
    """
    Sets material parameters
    """
    K=20.*10.**3.; G=10.*10.**3.; 
    tanfi=0.; cc=100.#*1000. #*0000000.;
    tanpsi=0.; Hsfi=0.; Hscc=-1;
    eta1=0.01
    
    prop_num=19
    props=np.zeros(prop_num)
    props[0]=K
    props[1]=G
    props[2]=0
    props[3]=0
    props[4]=0
    props[5]=0
    props[6]=0
    props[7]=0
    props[8]=0
    props[9]=0
    props[10]=0
    props[11]=cc
    props[12]=0
    props[13]=0
    props[14]=Hscc
    props[15]=0
    props[16]=0
    props[17]=0
    props[18]=eta1
    props=props.astype("double")
    return props

def p(stress):
    p=stress[0]+stress[1]+stress[2]
    print(stress[0],stress[1],stress[2])
    return p/3.

def q(stress):
#     q=stress[5]
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

def sigma_12(stress):
    sigma_12=stress[5]
    return sigma_12

def gamma_12(deformation):
    gamma_12=deformation[5]
    return gamma_12


class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''
        Run a load path for Drucker Prager Material
        '''
        cls.notfirsttime=True
        env_lib=ngeo_parameters.env_lib        #umat_lib='./libplast_Cauchy3D-DP.so'
        umat_lib_path= ngeo_parameters.umat_lib_path
        umat_lib = umat_lib_path+'CAUCHY3D-DP/libplast_Cauchy3D-DP.so'
        umat_id=1       # if many materials exist in the same library
        mat=UserMaterial(env_lib,umat_lib,umat_id)
        mat.props=set_material_1_properties()

        increments=100
#         e11min=0.;e11max=.1;deps11=(e11max-e11min)/increments
        e22min=0.;e22max=0.;deps22=(e22max-e22min)/increments
        e12min=0.;e12max=1.;deps12=(e12max-e12min)/increments
#
        deps=[np.array([0.,0.])]
        for i in range(increments):
            deps.append(np.array([deps22,deps12]))
                    
        with open('deps','w') as f:
            for current_strain in zip(deps):
                f.write("{0}\n" .format(str(current_strain)))
        
        stressGP_t=np.zeros(2)
        svarsGP_t=np.zeros(38)
        dsdeGP_t=np.zeros(2**2)
        dt=1.
        
        stress=[];ps=[];qs=[]
        epsilon=[];evs=[];eqs=[];sigma12s=[];gamma12s=[]
        stress_total=[]
        for i in range(len(deps)):
            #print("Increment: ",i)
            deGP=deps[i][:].copy()
            print(deGP)
            nill=mat.usermatGP(stressGP_t,deGP, svarsGP_t, dsdeGP_t, dt,0)
#             print(stressGP_t,"alex \n",svarsGP_t)
            if nill!=1:
                ps.append(p(svarsGP_t[0:12]))
                qs.append(q(svarsGP_t[0:12]))
                
                stress.append(svarsGP_t[0:12])
                stress_total.append(stressGP_t[0])
                
                evs.append(ev(svarsGP_t[12:24]))
                eqs.append(eq(svarsGP_t[12:24]))
                epsilon.append(svarsGP_t[12:24])
                sigma12s.append(sigma_12(svarsGP_t[0:12]))
                gamma12s.append(gamma_12(svarsGP_t[12:24]))
#                 print('svars')
#                 print(svarsGP_t)
#                 print('dsde')
#                 print(dsdeGP_t)
            else:
                print("material problem")
                return
        
        val_exact=[20000,11547.005038,1.,2.309,0.,1.]
        print(ps[-1],qs[-1],evs[-1],eqs[-1],sigma12s[-1],gamma12s[-1])
        
        with open('stress','w') as f:
            for current_stress in stress:
                f.write("%s\n" % str(current_stress))
#         print(len(stress),stress[0],stress[-1])
        #plt.rc('text', usetex=True)
        #plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
        #plt.ion() 
        plt.plot(ps, qs, "bo-", label="$p-q$")
        plt.xlabel("$p$")
        plt.ylabel("$q$")
#         plt.ylim(ymax=1.e2)
        plt.legend()
        plt.show()
         
        plt.plot(evs, ps, "bo-", label="$p-\epsilon_v$")
        plt.plot(evs, stress_total, "g+-", label="$\sigma_{22}-\epsilon_v$")
        plt.ylabel("$p$")
        plt.xlabel("$\epsilon_v$")
        plt.legend()
        plt.show()
         
        plt.plot(eqs, qs, "bo-", label="$q-\epsilon_q$")
        plt.ylabel("$q$")
#         plt.ylim(ymax=1.e2)
        plt.xlabel("$\epsilon_q$")
        plt.legend()
        plt.show()
        
        plt.plot(gamma12s ,sigma12s,  "bo-", label="$\sigma_{12}-\epsilon_{12}$")
        plt.ylabel("$\sigma_{12}$")
#         plt.ylim(ymax=1.e2)
        plt.xlabel("$\gamma_{12}$")
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
        Tests Drucker Prager material (stresses)
        '''
        values= np.array(self.stress)
#         with open("DP_stress_values.out", "wb") as fp:   #Pickling
#             pickle.dump(values, fp)
        reference_data_path = ngeo_parameters.reference_data_path    
        reference_data = reference_data_path+'DP_stress_values.out'
            
        with open(reference_data, "rb") as fp:   # Unpickling
            values_ref = pickle.load(fp)
             
        values_diff=values_ref-values
         
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
        reference_data = reference_data_path+'DP_epsilon_values.out'
        with open(reference_data, "rb") as fp:   # Unpickling
            values_ref = pickle.load(fp)
             
        values_diff=values_ref-values
         
        equal=abs(np.linalg.norm(values_diff))<=1.e-13
         
        self.assertTrue(equal, "Not identical deformations: "+str(abs(np.linalg.norm(values_diff))))
    

        
if __name__ == '__main__':
    unittest.main()