from typing import Union, Tuple, Dict, Any
from phi.torch.flow import *
from phiml.math import Tensor, Shape, extrapolation
from . import UniformGrid
from . import BaseBox
from ._geom import Geometry
from .. import math

class Voxels(Geometry):
    
    def __init__(self, grid: UniformGrid, filled: Tensor):
        self._grid = grid
        self._filled = filled

    
    @property
    def center(self) -> Tensor:
        return self._grid.center[self._filled]

    @property
    def shape(self) -> Shape:
        return self._filled.shape & self._grid.shape['vector']
    
    @property
    def resolution(self):
        return self._grid.resolution

    @property
    def volume(self) -> Tensor:
        raise NotImplementedError

    @property
    def faces(self) -> 'Geometry':
        raise NotImplementedError

    @property
    def face_centers(self) -> Tensor:
        raise NotImplementedError

    @property
    def face_areas(self) -> Tensor:
        raise NotImplementedError

    @property
    def face_normals(self) -> Tensor:
        raise NotImplementedError

    @property
    def boundary_elements(self) -> Dict[Any, Dict[str, slice]]:
        raise NotImplementedError

    @property
    def boundary_faces(self) -> Dict[Any, Dict[str, slice]]:
        raise NotImplementedError

    @property
    def face_shape(self) -> Shape:
        raise NotImplementedError

    def lies_inside(self, location: Tensor) -> Tensor:
        return (self._filled==1)
    
    def approximate_closest_surface(self, location: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def approximate_signed_distance(self, location: Tensor) -> Tensor:
        
        X, Y, Z = int(location.shape[0]), int(location.shape[1]), int(location.shape[2])
        
        ## Computing centered grid length using X,Y,Z since in all possible
        ## combinations of X,Y,Z one is L+1 and the rest are L
        L = (X+Y+Z)//3

        #self._filled is a cell centered grid of the shape (L,L,L) with solid,fluid locations marked
        # as 1,-1 respectively
        filled = self._filled
        filled_ = math.concat([ filled.x[:1].y[:].z[:] ,filled ,  filled.x[-1:].y[:].z[:]], dim=spatial('x'))
        filled_ = math.concat([filled_.x[:].y[:1].z[:], filled_, filled_.x[:].y[-1:].z[:]], dim=spatial('y'))
        filled_ = math.concat([filled_.x[:].y[:].z[:1], filled_, filled_.x[:].y[:].z[-1:]], dim=spatial('z'))
        
        #-------------------x face centered grid--------------------
        if location.shape.spatial == math.spatial(x=L+1,y=L,z=L):
            field_x = math.ones(math.spatial(x=L+1,y=L,z=L))

            # For computing x face locations that lie on solid-fluid interface
            obst_field_l = filled_.x[0:-1].y[1:-1].z[1:-1]
            obst_field_r = filled_.x[1:].y[1:-1].z[1:-1]

            # Mark the x faces that lie inside or on solid-fluid interfaces
            cond_x_in_at = (obst_field_l+obst_field_r>0)
            
            # Assign value -1 to faces that lie inside or on the interfaces
            # 1 to ones outside the interface
            field_x = math.where(cond_x_in_at,-1,field_x)
            cond_in_at = (field_x==-1)

            #field_x = torch.where(cond_x_at,  1,field_x)
            
            #print('field_x.x[25].y[:].z[:]')
            #math.print(math.tensor(field_x, spatial('x,y,z')).x[25].y[:].z[:])

            # Padding face centered grid field_x: (L+1, L, L) -> (L+1, L+2, L+2)
            field_x_ = math.concat([ field_x.x[:].y[:1].z[:], field_x,  field_x.x[:].y[-1:].z[:]],dim=spatial('y'))
            field_x_ = math.concat([field_x_.x[:].y[:].z[:1], field_x_,field_x_.x[:].y[:].z[-1:]],dim=spatial('z'))
            
            # Finally marking x faces whose cells are direct neighbours to obstacles cells as 1
            # If sum cond grid is equal to 1, the face belongs to a cell next to an obstacle cell
            cond =  (field_x_.x[:].y[ :-2].z[ :-2]==-1) + (field_x_.x[:].y[ :-2].z[1:-1]==-1) + (field_x_.x[:].y[:-2].z[2:]==-1) 
            cond += (field_x_.x[:].y[1:-1].z[ :-2]==-1) + (field_x_.x[:].y[1:-1].z[2:  ]==-1)
            cond += (field_x_.x[:].y[2:  ].z[ :-2]==-1) + (field_x_.x[:].y[2:  ].z[1:-1]==-1) + (field_x_.x[:].y[2:].z[2:]==-1)
            cond *= (field_x_.x[:].y[1:-1].z[1:-1]==1)

            loc = math.ones(math.spatial(x=L+1,y=L,z=L)) * 10.0
        #-----------------------------------------

        #-------------------y face centered grid--------------------
        elif location.shape.spatial == math.spatial(x=L,y=L+1,z=L):        
            field_y = math.ones(math.spatial(x=L,y=L+1,z=L))
            
            # For computing y face locations that lie on solid-fluid interface
            obst_field_u = filled_.x[1:-1].y[0:-1].z[1:-1]
            obst_field_d = filled_.x[1:-1].y[1:  ].z[1:-1]
            
            # Mark the x faces that lie inside or on solid-fluid interfaces
            cond_y_in_at = (obst_field_u+obst_field_d>0)

            # Assign value -1 to faces that lie inside or on the interfaces
            # 1 to ones outside the interface
            field_y = math.where(cond_y_in_at,-1,field_y)
            cond_in_at = (field_y==-1)

            # Padding face centered grid field_x: (L, L+1, L) -> (L+2, L+1, L+2)
            field_y_ = math.concat([ field_y.x[:1].y[:].z[: ], field_y,   field_y.x[-1:].y[:].z[  :]], dim=spatial('x'))
            field_y_ = math.concat([field_y_.x[ :].y[:].z[:1], field_y_, field_y_.x[  :].y[:].z[-1:]], dim=spatial('z'))
            
            # Finally marking y faces whose cells are direct neighbours to obstacles cells as 1
            # If sum cond grid is equal to 1, the face belongs to a cell next to an obstacle cell
            cond =  (field_y_.x[ :-2].y[:].z[0:-2]==-1) +  (field_y_.x[ :-2].y[:].z[1:-1]==-1) + (field_y_.x[:-2].y[:].z[2:]==-1) 
            cond += (field_y_.x[1:-1].y[:].z[ :-2]==-1) +  (field_y_.x[1:-1].y[:].z[2:  ]==-1)
            cond += (field_y_.x[2:  ].y[:].z[ :-2]==-1) +  (field_y_.x[2:  ].y[:].z[1:-1]==-1) + (field_y_.x[2:].y[:].z[2:]==-1)
            cond *= (field_y_.x[1:-1].y[:].z[1:-1]==1)

            loc = math.ones(math.spatial(x=L,y=L+1,z=L))  * 10.0


        #-----------------------------------------

        #-------------------z face centers--------------------
        elif location.shape.spatial == math.spatial(x=L,y=L,z=L+1):
            field_z = math.ones(math.spatial(x=L,y=L,z=L+1))
            
            # For computing y face locations that lie on solid-fluid interface
            obst_field_f = filled_.x[1:-1].y[1:-1].z[0:-1]
            obst_field_b = filled_.x[1:-1].y[1:-1].z[1:  ]

            # Mark the x faces that lie inside or on solid-fluid interfaces 
            cond_z_in_at = (obst_field_f+obst_field_b>0)
            #cond_z_at = (obst_field_f+obst_field_b>0) * (obst_field_f+obst_field_b<2) 
            #cond_x_out= not cond_x

            # Assign value -1 to faces that lie inside or on the interface,
            # 1 to ones outside the interface
            field_z = math.where(cond_z_in_at,-1,field_z)
            cond_in_at = (field_z==-1)

            # Padding face centered grid field_z: (L, L, L+1) -> (L+2, L+2, L+1)
            field_z_ = math.concat([ field_z.x[:1].y[: ].z[:], field_z, field_z.x[-1:].y[:  ].z[:]], dim=spatial('x'))
            field_z_ = math.concat([field_z_.x[ :].y[:1].z[:],field_z_,field_z_.x[  :].y[-1:].z[:]],dim=spatial('y'))
            
            # Finally marking z faces whose cells are direct neighbours to obstacles cells as 1
            # If sum cond grid is equal to 1, the face belongs to a cell next to an obstacle cell
            cond  = (field_z_.x[ :-2].y[0:-2].z[:]==-1) + (field_z_.x[ :-2].y[1:-1].z[:]==-1) + (field_z_.x[:-2].y[2:].z[:]==-1) 
            cond += (field_z_.x[1:-1].y[ :-2].z[:]==-1) + (field_z_.x[1:-1].y[2:  ].z[:]==-1)
            cond += (field_z_.x[2:  ].y[ :-2].z[:]==-1) + (field_z_.x[2:  ].y[1:-1].z[:]==-1) + (field_z_.x[2:].y[2:].z[:]==-1)
            cond *= (field_z_.x[1:-1].y[1:-1].z[:]==1)

            loc = math.ones(math.spatial(x=L,y=L,z=L+1)) * 10.0
            
           
        #------------------------------------------
        ## loc=1 for face centers whose cells are next to solid-fluid interface
        loc = math.where(cond, 1, loc)
        ## loc=0.0 for face centers whose cells are part of the solid
        loc = math.where(cond_in_at, 0, loc)


        distance = math.ones(location.shape.spatial) * 10.0
        distance = math.where( (loc==1), 0.5, distance)
        
        distance = math.where(loc==0 , 0, distance)

        distance = math.tensor(distance, spatial('x,y,z'))

        distance = math.where(distance>0.6, 10, distance)
        distance = math.where(distance<=0, 0, distance)

        return distance

    
    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError

    def bounding_radius(self) -> Tensor:
        raise NotImplementedError

    def bounding_half_extent(self) -> Tensor:
        raise NotImplementedError

    def at(self, center: Tensor) -> 'Geometry':
        raise NotImplementedError

    def __variable_attrs__(self):
        return '_grid', '_filled'
    
    def __value_attrs__(self):
        return '_grid', '_filled'

    
    def rotated(self, angle: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError