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
#     def __init__(self, file="",silent=True):
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
#         self.O.trackEnergy=True # enable energy tracking   
#         self.O.cell.trsf=yd.Matrix3.Identity # set the current cell configuration to be the reference one to compute strain
#         self.O.dynDt=True # enable adaptive critical time step calculation in yade (True by default, but fdor being sure)
#         # Other Parameters
#         self.run_increment_in_background=True # runs yade increment in background
#         self.substeppingfactor=1.1
#         self.nparticles=self.get_number_of_particles()
#         self.filename=file
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
#     def doincrement(self,de,Delta_t=1.):
#         self.O.cell.velGrad=self.Voigt_to_Tensor(de,strain=True)/Delta_t # valid for infinitesimal strains
#         self.t_analysis=self.O.time
#         t_target=self.t_analysis+Delta_t
#         # yade O.stopAttime has a bug in yade and it doesn't work (01/2021) 
#         while t_target>self.O.time:
#             dt_crit=self.O.dt
#             nsteps=int(((t_target-self.O.time)/dt_crit)/self.substeppingfactor)
#             if nsteps==0: #complete the last increment 
#                 self.O.dt=t_target-self.O.time
#                 nsteps=1
# #             print("asdasdsadsadas",self.silent)
#             if self.silent==False: print("Executing:",nsteps," timesteps")
#             self.O.run(nSteps=nsteps,wait=self.run_increment_in_background)
#             self.O.dt=dt_crit
#
#         self.t_analysis=self.O.time
#         if self.t_analysis>t_target:
#             if self.silent==False: print("warning: targe time was exceeded by:",self.t_analysis-t_target, "last time step increment of the DEM analysis was",self.O.dt)
#         return self.t_analysis/Delta_t # ratio of target epsilon achieved
#
#     def get_sym_tensor(self,tensor):
#         return .5*(tensor + tensor.transpose())
#
#     def output_increment_data(self):
#         # get the stress tensor (as 3x3 matrix)
#         stress_tensor=self.get_sym_tensor((np.array(yd.utils.getStress())))  # stress tensor
#         F = self.O.cell.trsf # transformation tensor
#         eps_tensor = self.get_sym_tensor(np.array(0.5*(F.transpose() * F - yd.Matrix3.Identity))) # deformation tensor
#         vol = self.O.cell.volume
#         E_total = self.O.energy.total()/vol
#         E_elast = self.O.energy['elastPotential']/vol
#         E_dissipated = self.O.energy['plastDissip']/vol #dissipation is total
#         time=self.O.time
#         res=np.array([(b.state.pos,b.state.vel) for b in self.O.bodies if isinstance(b.shape,yd.Sphere)]) #,b.shape.radius
#         xs=res[:,0].flatten()
#         vs=res[:,1].flatten()
#         return time, self.Tensor_to_Voigt(stress_tensor), self.Tensor_to_Voigt(eps_tensor,True), E_elast, E_dissipated, E_total, yd.utils.unbalancedForce(), xs, vs 
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
#         tensor=np.asarray([vector[0],       mult*vector[5], mult*vector[4],
#                        mult*vector[5],  vector[1],      mult*vector[3],
#                        mult*vector[4],  mult*vector[3], vector[2]],
#                        dtype=np.float64)
#         tensor=tensor.reshape((-1,3))
#         return tensor
#
#     def Tensor_to_Voigt(self, tensor, strain=False):
#         mult=1.
#         if strain==True: mult=2.
#         voigt=np.array([tensor[0,0],tensor[1,1],tensor[2,2],mult*tensor[1,2],mult*tensor[0,2],mult*tensor[0,1]])
#         return voigt