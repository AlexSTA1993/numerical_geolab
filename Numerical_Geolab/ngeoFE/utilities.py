'''
Created on Aug 23, 2018

@author: Ioannis Stefanou

Provides general utilities for ngeoFE package. 
'''
# from dolfin.cpp.mesh import MeshFunction, Mesh
from dolfin import *
from dolfin.cpp.io import HDF5File

from operator import itemgetter
import math
#
def Convert_mesh_xml_to_hdmf(xmlfile,xmlfacesfile,xmlregionsfile,hdmfoutputfile):
    '''
    Converts xml dolfin mesh to hdmf format which allows parallel computations.

    :param xmlfile: xml input filename
    :type xmlfile: string
    :param xmlfacesfile: xml facets input filename
    :type xmlfacesfile: string
    :param xmlregionsfile: xml internal regions input filename
    :type xmlregionsfile: string
    :param hdmfoutputfile: hdmf output filename
    :type hdmfoutputfile: string
    '''
    mesh = Mesh(xmlfile);
    cd=MeshFunction('size_t',mesh,xmlregionsfile);
    fd=MeshFunction('size_t',mesh,xmlfacesfile);
    hdf = HDF5File(mesh.mpi_comm(), hdmfoutputfile, "w")
    hdf.write(mesh, "/mesh")
    hdf.write(cd, "/cd")
    hdf.write(fd, "/fd")

    #def map_ncoor_dofval(self,b_nodal,function_u):
def get_u_coordinates(function_u,V,mesh):


    dofmap = V.dofmap()
    dofs = dofmap.dofs()
    gdim = mesh.geometry().dim()
    # Get coordinates as len(dofs) x gdim array
    dofs_x = V.tabulate_dof_coordinates().reshape((-1, gdim))

    w=function_u.vector().get_local()
#     print(dofs_x)
    values=[]
    for dof, dof_x, wi in zip(dofs, dofs_x, w):
        line=[dof,dof_x,wi]
        print(line)
        values.append(line)
#         print(dof, ':', dof_x, "u=",wi)

    return values


#####SCRAP


#     class MyFunctionExpression(UserExpression):
#         def eval(self, values, x):
#             values[0] = 0.
#             values[1] = 0.
#             if near(x[0],-1./2.): values[0]=0.01
#             if near(x[1],1./2.): values[1]=-0.01
#         def value_shape(self):
#             return (2,)
#         
#     def testexpr(self):
#         f = self.MyFunctionExpression(element=self.V.ufl_element())
#         self.ti=interpolate(f,self.V)
#         print("hello0000 region d ti=  ",self.ti.vector().get_local())        

#                 #[l.extend([[k,v]]) for k,v in dofs_values.items()]
#                 print("hello4444",self.comm.Get_rank(),">",lc_dofs)
#                               
#                 # works in parallel, as boundaries is a parallel vector and each cpu has its vertices (and ghost vertices).
#                 itemindex=self.boundaries.where_equal(region_id)
#                 
#                 for entity in itemindex:
#                     #print("xxxxxxxxxx",self.comm.Get_rank(),">",MeshEntity(self.mesh,self.boundaries.dim(),entity).dofs())
#                     
#                     mesh_entity=MeshEntity(self.mesh, self.boundaries.dim(), entity)
#                     print("kkkkkkkkkkkkkkk",mesh_entity,mesh_entity.index(),mesh_entity.global_index())
#                     #for cell in cells(self.mesh):
#                     for cell in cells(mesh_entity):
#                         print("lllllllllllllllllllllllllllllll",cell,cell.index(),cell.global_index())
#                         #print("normal",cell.index(),)
#                         #node_coordinates = dofmap.tabulate_coordinates(cell)
#                         global_dofs = dofmap.cell_dofs(cell.global_index())
#                         #print('node coordinates:',node_coordinates)
#                         #print('dofs global:',self.comm.Get_rank(),">",global_dofs)
#                         for gl_dof in global_dofs:
#                              
#                             lc_dof=np.where(lc_to_gl_dof_owned==gl_dof)
#                             #print("hello4",self.comm.Get_rank(),">",lc_dof,np.shape(lc_dof),len(lc_dof))
#                             if not 0 in np.shape(lc_dof): dof_ids.append(lc_dof[0][0])

                        #local_dofs=np.array([gl_to_lc_owned_dof(gl_dof) for gl_dof in global_dofs ]).flatten()
                        #print('dofs local:',local_dofs)
                        #local_dofs=global_dofs
                        #dof_ids.append(local_dofs)




#                     
#                     for vx in vertices(MeshEntity(self.mesh,self.boundaries.dim(),entity)):
#                         print("dffsdfsafasdf",self.comm.Get_rank(),">",vx.index(),v_to_dof_gl[vx.index()])
#                           
#                         gl_dof_index=v_to_dof_gl[vx.index()]
#                         if gl_dof_index>= process_gl_dofs_range_min and gl_dof_index <= process_gl_dofs_range_max:
#                         #print("hello2",self.comm.Get_rank(),">",vx.index(),vx.global_index(),self.V.dofmap().local_to_global_index(vx.index()),vx.is_shared())
# #                             print("hello4",self.comm.Get_rank(),">",vx.index()) 
#                             if vx.is_shared()==True: print("shared",self.comm.Get_rank(),">",vx.global_index())
#                             dof_ids.append(v_to_dof_local[vx.index()])
                #lc_dofs=np.unique(dof_ids)




#         top0= AutoSubDomain(lambda x, on_bnd: x[1]>+1./2.-DOLFIN_EPS and on_bnd)
#         
#         bmesh=BoundaryMesh(self.mesh,"exterior")
#         bfunction=MeshFunction("size_t", bmesh, bmesh.topology().dim())
#         top0.mark(bfunction,1)
# 
#         
#         ttt=Function(self.V) 
#         
#         E = MyExpr2D(self.boundaries)
#         
#         ttt.interpolate(E)




#         v_to_dof_local=vertex_to_dof_map(self.V)
#         
#         
#         v_to_dof_gl2=list(self.V.dofmap().local_to_global_index(v_to_dof_local[i]) for i in v_to_dof_local)

#         lc_to_gl_dof=dofmap.tabulate_local_to_global_dofs()
#         print("0000000000",self.comm.Get_rank(),">",dofmap.dofs(self.mesh,0),dofmap.dofs())
# 
#         unowned_gl_dof=dofmap.local_to_global_unowned()
#         
#         
#         
#         print("hello",self.comm.Get_rank(),">",self.V.dofmap().off_process_owner())
# #         
# 
#         print("hello1",self.comm.Get_rank(),">",lc_to_gl_dof)        
#         lc_to_gl_dof_owned=np.setdiff1d(lc_to_gl_dof,unowned_gl_dof)
#         print("hello2",self.comm.Get_rank(),">",lc_to_gl_dof_owned)
#         
#         #gl_to_lc_owned_dof=lambda gl_dof: np.where(lc_to_gl_dof_owned==gl_dof)
#         
#         print("hello2",self.comm.Get_rank(),">",lc_to_gl_dof)

#         v_to_dof_gl=list(lc_to_gl_dof[i] for i in v_to_dof_local)
# 
#         print("hello1-",self.comm.Get_rank(),">",v_to_dof_gl)
#         print("hello1+",self.comm.Get_rank(),">",v_to_dof_gl2)
#         print("hello1",self.comm.Get_rank(),">",vertex_to_dof_map(self.V))
#         print("hello2",self.comm.Get_rank(),">",dof_to_vertex_map(self.V))
#         
#         
#         print("hello5",self.comm.Get_rank(),">",self.V.dofmap().tabulate_local_to_global_dofs())
#         #print("hello6",self.comm.Get_rank(),">",self.V.dofmap().tabulate_entity_dofs())
#         
#         
#         process_gl_dofs_range_min=min(self.V.dofmap().ownership_range())
#         process_gl_dofs_range_max=max(self.V.dofmap().ownership_range())
#          
#         print("hello3",self.comm.Get_rank(),">","min dof",process_gl_dofs_range_min,"max dof",process_gl_dofs_range_max)


                    #lc_dof=np.where(lc_to_gl_dof_owned==gl_dof)
                    #print("hello4",self.comm.Get_rank(),">",lc_dof,np.shape(lc_dof),len(lc_dof))
                    #if not 0 in np.shape(lc_dof):
                                         #   lc_dof_values.append(value)
                #lc_dofs=lc_dofs[:len(lc_to_gl_dof)]

                #print("hello2000000000",self.comm.Get_rank(),">",lc_dofs)
#                 
#                 a=Vector()
#                 a.init(10)
#                 print(1)
#                 
#                 e=a.get_local()
#                 print("RRRRRRRRRRRRR",e[:])

#                 local_dofs_values=BC.get_boundary_values() #returns in local....... perversity
#                 lc_dofs=[];
#                 for lc_dof in local_dofs_values.keys():
#                     if lc_dof<=len(lc_to_gl_dof): lc_dofs.append(lc_dof)

# def map_ncoor_dofval(function_u,V,mesh):
# 
#     
#     
#         list_b=[]
#         u_new=[]
# #         gamma_new=[]
#         for i in range(len(self.function_u)):
#             #print(len(self.function_u))
#             u_new.append(self.function_u[i])
#         
#         for i in range(len(self.dofs_x)):
#             list_b.append([self.dofs_x[i],b_nodal[i],u_new[i]])
# #         print("hello",list_b)
#         
#         
#         list_ncoor_dofs=[]
#         q=0
#         m=-1
# #q1=0      
#         for i in range(len(list_b)):
#             if list_b[i]!=None:
#                 m+=1
#                 list_ncoor_dofs.append([])
#                 #list_ncoor_dofs.append([list_b[i][0]])
#                 #list_ncoor_dofs[m].append(list_b[i][1])
#                 
#                 for nel in range(3):    #number of functions to take output
#                     list_ncoor_dofs[m].append([])
#                     #print("hello", list_ncoor_dofs[m])
#         
#                 for nel in range(3):    #number of functions to take output
#                     list_ncoor_dofs[m][nel].append(list_b[i][nel])
#                     #print(list_ncoor_dofs[m][nel])
#                 
#                 q=i+1
#                 for j in range(q,(len(b_nodal))):
#                     if list_b[j]!=None:
#                         if list_b[i][0]==list_b[j][0]:
#                             if list_b[i]!=None:
#                                 list_ncoor_dofs[m][1].append(list_b[j][1])
#                                 list_ncoor_dofs[m][2].append(list_b[j][2])
#                                 list_b[j]=None
#         list_ncoor_dofs=sorted(list_ncoor_dofs,key=itemgetter(0))
# #         print("hello_list",list_ncoor_dofs)
# #         print(len(self.dofs_x))
#         
#         
#             
#          
#        
#         ncoor_dof={}
#         for i in range(len(list_ncoor_dofs)):
#             listnew=[]
# #             print(i)
# #             print(list_ncoor_dofs[i])
# #             print(len(list_ncoor_dofs[i]))
# #             print((len(self.dofs_x)/len(list_ncoor_dofs)))
#             for j in range(0,math.ceil(len(self.dofs_x)/len(list_ncoor_dofs))):
#                 listnew.append(list_ncoor_dofs[i][j])
# #                print(listnew)
#             #print(listnew)
#             #ncoor_dof[list_ncoor_dofs[i][0]]=listnew  
#             ncoor_dof[i]=listnew              
# #         print(ncoor_dof)
#         return ncoor_dof
#     
#     
