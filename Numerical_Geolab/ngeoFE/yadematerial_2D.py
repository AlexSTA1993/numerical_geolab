# '''
# Created on Jan 14, 2021
#
# @author: Ioannis Stefanou & Filippo Masi
# '''
# import yade as yd
# import numpy as np
#
# class YadeMaterial():
#     """
#     YADE material class
#     """
#     ##TODO: Messages with silent or not message IO of nGeo
#     def __init__(self, file="", silent=True):
#         self.silent=silent
#         if file == "" :
#             self.O = self.generate_sample()
#         else:
#             self.O = self._generate_sample_from_file(file)
#             if self.silent==False: print("Sample successfully loaded")
#         if self.O == None:
#             print("Error in loading sample - exiting")
#             return
#
#         # General parameters for the DEM analysis
#         #self.O.trackEnergy=True # enable energy tracking   
#         self.O.cell.trsf=yd.Matrix3.Identity # set the current cell configuration to be the reference one to compute strain
#         self.O.dynDt=True # enable adaptive critical time step calculation in yade (True by default, but fdor being sure)
#         # Other Parameters
#         self.run_increment_in_background=True # runs yade increment in background
#         self.substeppingfactor=1.1
#         self.nparticles=self.get_number_of_particles()
#         self.filename=file
#
#
#     def reset(self):
#         self.__init__(self.filename,self.silent)
#
#     def generate_sample(self):
#         if self.silent==False: print("No code provided for in creating sample")
#         return
#
#     def _generate_sample_from_file(self, file):    
#         yd.O.reset() 
#         yd.O.load(file)
#         yd.O.resetTime() # resets time of the anaysis
#         return yd.O
#
#
#     def doincrement(self,de,loading_delta_t=1.):
#         self.O.cell.velGrad=self.Voigt_to_velGrad(de,strain=True)/loading_delta_t
# #         self.O.cell.velGrad=self.Voigt_to_Tensor(de,strain=True)/loading_delta_t # valid for infinitesimal strains
#         self.t_analysis=self.O.time
#         t_target=self.t_analysis+loading_delta_t
# #         print(loading_delta_t)
#         # yade O.stopAttime has a bug in yade and it doesn't work (01/2021) 
#         while t_target>self.O.time:
# #             print(self.O.dt)
#             dt_crit=self.O.dt
#             nsteps=int(((t_target-self.O.time)/dt_crit)/self.substeppingfactor)
#             if nsteps==0: #complete the last increment  can be problematic cause you re forcing Z to have dtc_crit
#                 self.O.dt=t_target-self.O.time
#                 nsteps=1
#             if self.silent==False: print("Executing:",nsteps," timesteps")
#             self.O.run(nSteps=nsteps,wait=self.run_increment_in_background)
#             self.O.dt=dt_crit
#
#         self.t_analysis=self.O.time
#         if self.t_analysis>t_target:
#             if self.silent==False: print("warning: targe time was exceeded by:",self.t_analysis-t_target, "last time step increment of the DEM analysis was",self.O.dt)
#         return self.t_analysis/loading_delta_t # ratio of target epsilon achieved
#
#     def get_sym_tensor(self,tensor):
#         return .5*(tensor + tensor.transpose())
#
#     def output_increment_data(self):
#         # get the stress tensor (as 3x3 matrix)
#         stress_tensor=self.get_sym_tensor((np.array(yd.utils.getStress())))  # stress tensor v
#         #### A FAIRE put controller on traction, sigma_ii < 0 always!
#         #### si statique, stress_tensor doit etre symmetrique! A verifier!
#         #### Une possibilite est verifier que le tenseur soit symmetrique et sinon augmente dt!
#
#         F = self.O.cell.trsf # transformation tensor
#         eps_tensor = (np.array(0.5*(F.transpose() * F - yd.Matrix3.Identity)))
#         vol = self.O.cell.volume
#         E_total = self.O.energy.total()/vol
#         E_elast = self.O.energy['elastPotential']/vol
#         E_dissipated = self.O.energy['plastDissip']/vol #dissipation is total
#         time=self.O.time
# #         print(np.array(yd.utils.getStress()))
#         s=np.array([(b.state.pos[0],b.state.pos[1],b.state.pos[2],b.state.vel[0],b.state.vel[1],b.state.vel[2],b.shape.radius) for b in self.O.bodies if isinstance(b.shape,yd.Sphere)])
#         xs = np.concatenate((s[:,0],s[:,1],s[:,2]),axis=0)
#         vs = np.concatenate((s[:,3],s[:,4],s[:,5]),axis=0)
# #         print(s[:10,2])
#         rad = s[:,6]
# #         
#         rot=np.array([(b.state.rot()) for b in self.O.bodies if isinstance(b.shape,yd.Sphere)])
#
#         print(time, self.Tensor_to_Voigt(stress_tensor).shape)
#         print(self.Tensor_to_Voigt(eps_tensor,True).shape)
#         print(yd.utils.unbalancedForce())
#         print(xs.shape)
#         print(vs.shape)
#         print(rad.shape)
#         print(rot.shape)
# #         res=np.array([(b.state.pos,b.state.vel) for b in self.O.bodies if isinstance(b.shape,yd.Sphere)]) #,b.shape.radius
# #         xs=res[:,0].flatten()
# #         vs=res[:,1].flatten()
# #         print(res[2,0],res[2,1],res[2,2])
#         return time, self.Tensor_to_Voigt(stress_tensor), self.Tensor_to_Voigt(eps_tensor,True), E_elast, E_dissipated, E_total, yd.utils.unbalancedForce(), xs,vs,rad#,rot
#
#     def get_number_of_particles(self):
#         n=0
#         for b in self.O.bodies:
#             if isinstance(b.shape,yd.Sphere):
#                 n+=1
#         return n
#
#     def Voigt_to_Tensor(self, vector, strain=False):
#         mult=1.
#         if strain==True: mult=.5
#         tensor=np.asarray([vector[0],   mult*vector[2],
#                            mult*vector[2],  vector[1]],
#                             dtype=np.float64)
#         tensor=tensor.reshape((-1,2))
#         return tensor
#
#     def Voigt_to_velGrad(self, vector, strain=False):
#         mult=1.
#         if strain==True: mult=.5
#         tensor=np.asarray([vector[0],   mult*vector[2], 0,
#                            mult*vector[2],  vector[1], 0,
#                            0,                        0, 0],
#                             dtype=np.float64)
#         tensor=tensor.reshape((-1,3))
#         return tensor
#
#     def Tensor_to_Voigt(self, tensor, strain=False):
#         mult=1.
#         if strain==True: mult=2.
#         voigt=np.array([tensor[0,0],tensor[1,1],mult*tensor[0,1]])
#         return voigt