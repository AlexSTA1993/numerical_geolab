'''
Created on Nov 10, 2020

@author: Ioannis Stefanou & Filippo Masi & Alexandros Stathas
'''

# Force use CPU only.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import numpy as np
import time

class AIUserMaterial():
    """
    AI Material class
    """
    def __init__(self, TANN_folder):
        """
        Load ANN network
        
        :param ANN_filename: ANN filename with path
        :type umat_lib: string
        """
        # Load ANN network
        self.model = tf.keras.models.load_model(TANN_folder)
#         print(self.model.summary())
        # TODO put if for error hjandling if noi loaded
        self.loaded=True
    
    def predict_wrapper(self,de,svars_t):
        """
        User material at a Gauss point
        
        :param deGP: generalized deformation vector at GP - input
        :type deGP: numpy array
        :param svarsGP_t: state variables at GP - input/output
        :type svarsGP_t: numpy array
        
        :return: generalized stress at GP output, state variables at GP - output, jacobian at GP - output
        :rtype: numpy array, numpy array, numpy array
        """
        input=np.concatenate((np.expand_dims(svars_t[:,0], axis=1),np.expand_dims(de, axis=1),svars_t[:,1:3]),1)
#         input=np.concatenate((svars_t[:,0],de,svars_t[:,1:2]))
        sigma_tdt, svars_tdt, Jac_tdt = self.model.predict(input)#(np.expand_dims(deGP, axis=0),np.expand_dims(svarsGP_t, axis=0))        
        return sigma_tdt, svars_tdt, Jac_tdt
    
    def usermatGP(self,stressGP_t,deGP,svarsGP_t,dsdeGP_t,dt,GP_id,aux_deGP=np.zeros(1)):
        """
        User material at a Gauss point
        
        :param stressGP_t: generalized stress at GP - input/output
        :type stressGP_t: numpy array
        :param deGP: generalized deformation vector at GP - input
        :type deGP: numpy array
        :param aux_deGP: auxiliary generalized deformation vector at GP - input
        :type aux_deGP: numpy array
        :param svarsGP_t: state variables at GP - input/output
        :type svarsGP_t: numpy array
        :param dsdeGP_t: jacobian at GP - output
        :type dsde_t: numpy array
        :param dt: time increment
        :type dt: double
        :param GP_id: Gauss Point id (global numbering of all Gauss Points in the problem) - for normal materials is of no use
        :type GP_id: integer
        :return: 0 if ok, 1 if failed
        :rtype: integer
        """       
#         inputs=np.array([inputs])
#         inputs=np.expand_dims(inputs, axis=0)
        tic = time.perf_counter()
#         print(self.model.predict(inputs))
        sigma_tdt, svars_tdt, Jac_tdt = self.predict_wrapper(deGP,np.expand_dims(svarsGP_t, axis=0))#(np.expand_dims(deGP, axis=0),np.expand_dims(svarsGP_t, axis=0))
#         if GP_id==12 or GP_id==13 or GP_id==14 or GP_id==15 or GP_id==16 or GP_id==17: Jac_tdt=np.array([[-10523.41751543]])#print(Jac_tdt)#=np.array([[200*10**3]])
#         alpha_tdt, sigma_tdt, Jac_tdt, F_tdt, D_tdt = self.model.predict(inputs)
        toc = time.perf_counter()
#         print("ḧello", Jac_tdt,sigma_tdt)
#         print(f"time {toc - tic:0.4f} seconds")
        svarsGP_t[:]=svars_tdt
        stressGP_t[:]=sigma_tdt
        dsdeGP_t[:]=Jac_tdt
        nill=0
        return nill
        
        
#         sigma_t=svarsGP_t[:model.nsigma]
#         epsilon_t=svarsGP_t[model.nsigma:2*model.nsigma]
#         alpha_t=svarsGP_t[2*model.nsigma:]
#         epsilon_tdt=epsilon_t+deGP
#         stress_tdt,alpha_tdt,jac,F,D = self.model.predict(epsilon_tdt,deGP,sigma_t,alpha_t)
#         alpha_tdt=alpha_t+d_alpha.squeeze()
#         sigma_tdt=sigma_t+d_stress.squeeze()
#         dsdeGP_t=jac.squeeze()
#         svarsGP_t=np.array([sigma_tdt,epsilont_dt,alpha_tdt]).flatten()
#         return nill


# ann_folder="/home/ist/Downloads/TANN1D_Model"
# annmat=AIUserMaterial(ann_folder)
