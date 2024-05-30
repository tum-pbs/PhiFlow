from phi.torch.flow import *
from phi.geom._voxels import Voxels
from phi.geom import UniformGrid


from phi.torch.flow import *


#convert to phiflow/math tensor (from torch tensor)
def to_phi_t(x, dim=3):
    if dim==3: return math.tensor(x, spatial('x,y,z'))
    elif dim==2: return math.tensor(x, spatial('x,y'))

#convert to torch tensor from (phiflow/math tensor)
def to_torch_t(x, dim=3):
    if dim==3: return math.reshaped_native(x, groups=['x','y', 'z'])
    elif dim==2: return math.reshaped_native(x, groups=['x','y'])



def construct_orifice3d(x, loc, r, t=2):
    x_range, y_range, z_range = x.shape
    reshape_tuple = (0,1,2)

    if x_range == t:
        x = x.permute(1,2,0)
        reshape_tuple = (2,0,1)
    elif y_range==t:
        x = x.permute(0,2,1)
        reshape_tuple = (0,2,1) 

    i_ = torch.arange(x.shape[0])
    j_ = torch.arange(x.shape[1])
    grid_i, grid_j = torch.meshgrid(i_, j_, indexing='ij')
    true_idxs = torch.nonzero((grid_i - loc[0])**2 + \
                              (grid_j - loc[1])**2 < r**2)
    x[true_idxs[:,0], true_idxs[:,1],:] = 1

    return x.permute(reshape_tuple)
                
def construct_orifice2d(x, loc, r, t=2):
    x_range, y_range = x.shape
    reshape_tuple = (0,1)

    if x_range == t:
        x = x.permute(1,0)
        reshape_tuple = (1,0) 
    
    true_idxs = torch.arange(loc, loc+2*r)
    x[true_idxs,:] = 1

    return x.permute(reshape_tuple)

def create_geometry2D(params):
    '''
    Takes 2D simulation case parameters as input and generates wall geometry and specifies boundary condition masks
    '''

    t = int(params['t'])
    res = int(params['topopt_domain_res'])
    X = res + 2*t
    Y = res + 2*t

    start_x, start_y = t, t
    end_x, end_y = res+t, res+t
    DOMAIN = dict(x=X, y=Y, bounds=Box(x=X, y=Y))

    domain_grid = CenteredGrid(0, **DOMAIN)

    D = params['topopt_domain_res']/10

    # Constructing Wall Obstacles
    
    INLET_MASK = None
    OUTLET_MASK = None
    BOUNDARY_MASK = None
    VEL_BOUNDARY = None
    wall_tensor = {}

    wall_tensor['left'] = torch.zeros(t,Y)
    wall_tensor['right'] = torch.zeros(t,Y)

    wall_tensor['bottom'] = torch.zeros(X,t)
    wall_tensor['top'] = torch.zeros(X,t)

    for i,loc in enumerate(params['orifice_locs']):
        wall_type = params['orifice_walls'][i]
        
        R = int(D*params['orifice_widths'][i]/2)
        #loc_ = [int(loc[0]*D)+t, int(loc[1]*D)+t]
        loc_ = int((loc-1) * D) + t
        wall_tensor[wall_type] = construct_orifice2d(wall_tensor[wall_type], loc_, r=R, t=t)  

    wall_types = ['left', 'bottom', 'right', 'top']

    
    # ## Adding Wall Obstacles
    num_obstacles = 0
    OBS_WALL_GEOMETRY = []
    obs_coords_list = []
    
    idxs_ = None
    for wall_type in wall_types:
        Y_range = range(0,Y)
        X_range = range(0,X)

        offset_x = 0
        offset_y = 0
        if wall_type == 'left':   
            X_range = range(0,start_x)
        elif wall_type == 'right':  
            X_range = range(end_x, X)
            offset_x = end_x
        elif wall_type == 'bottom': 
            Y_range = range(0,start_y)
        elif wall_type == 'top':
            Y_range = range(end_y, Y)
            offset_y = end_y

        
        idxs = torch.nonzero(wall_tensor[wall_type] == 0)
        
        idxs += torch.tensor([offset_x, offset_y]).unsqueeze(0)
        if idxs_ == None:
            idxs_ = idxs
        else:
            idxs_ = torch.cat([idxs_, idxs], dim=0)

        ## Voxel not implemented for 2D yet
        for x_val in X_range:
            for y_val in Y_range:
                if wall_tensor[wall_type][x_val-offset_x, y_val-offset_y]==0 and \
                    (x_val, y_val) not in obs_coords_list:
                        OBS_WALL_GEOMETRY.append(Box['x,y', x_val:x_val+1, y_val:y_val+1])
                        obs_coords_list.append((x_val, y_val))
                        num_obstacles += 1
    #OBS_WALL_GEOMETRY = Voxels(UniformGrid(resolution=domain_grid.shape, bounds=Box(x=X, y=Y, z=Z)), obst_grid_)
    

    inlet_walls = []
    for i, vel in enumerate(params['inlet_velocities']):
        if vel!=0:
            if params['orifice_walls'][i] not in inlet_walls:
                inlet_walls.append(params['orifice_walls'][i])
        else:
            outlet_wall = params['orifice_walls'][i]
    
    wall_mask = {}
    wall_maskp = {}
    wall_vec = {}
    wall_mask['left'] = math.concat([to_phi_t(wall_tensor['left'][0,:].unsqueeze(0), dim=2),
                                 to_phi_t(torch.zeros(X-1,Y), dim=2)], dim=spatial('x'))
    wall_mask['right'] =math.concat([to_phi_t(torch.zeros(X-1,Y), dim=2), 
                                  to_phi_t(wall_tensor['right'][0,:].unsqueeze(0), dim=2)], dim=spatial('x'))
    wall_mask['bottom']=math.concat([to_phi_t(wall_tensor['bottom'][:,0].unsqueeze(1), dim=2),
                                  to_phi_t(torch.zeros(X,Y-1), dim=2)], dim=spatial('y'))
    wall_mask['top']   =math.concat([to_phi_t(torch.zeros(X,Y-1), dim=2)
                                 ,to_phi_t(wall_tensor['top'][:,0].unsqueeze(1), dim=2)], dim=spatial('y'))
    
    wall_vec['left'] = (1,0)
    wall_vec['right']= (-1,0)
    wall_vec['bottom']=(0,1)
    wall_vec['top']   =(0,-1)

    wall_maskp['left'] = math.concat([to_phi_t(torch.zeros(t-1,Y), dim=2), to_phi_t(wall_tensor['left'][0,:].unsqueeze(0), dim=2),
                              to_phi_t(torch.zeros(X-t,Y), dim=2)], dim=spatial('x'))

    wall_maskp['right'] =math.concat([to_phi_t(torch.zeros(X-t,Y), dim=2), 
                               to_phi_t(wall_tensor['right'][0,:].unsqueeze(0), dim=2),to_phi_t(torch.zeros(t-1,Y), dim=2) ], dim=spatial('x'))

    wall_maskp['bottom']=math.concat([to_phi_t(torch.zeros(X, t-1), dim=2),  to_phi_t(wall_tensor['bottom'][:,0].unsqueeze(1), dim=2),
                               to_phi_t(torch.zeros(X,Y-t), dim=2)], dim=spatial('y'))

    wall_maskp['top']   =math.concat([to_phi_t(torch.zeros(X,Y-t), dim=2)
                              ,to_phi_t(wall_tensor['top'][:,0].unsqueeze(1), dim=2), to_phi_t(torch.zeros(X, t-1), dim=2)], dim=spatial('y'))


    visited_walls = []
    for i, vel in enumerate(params['inlet_velocities']):
        wall = params['orifice_walls'][i]
        wall_mask_ = wall_mask[wall]
        wall_maskp_ = wall_maskp[wall]
        if abs(vel)>0:
            if INLET_MASK == None:
                INLET_MASK = CenteredGrid(wall_maskp_, ZERO_GRADIENT, **DOMAIN)
            elif wall not in visited_walls:
                INLET_MASK += CenteredGrid(wall_maskp_, ZERO_GRADIENT, **DOMAIN)
            if VEL_BOUNDARY==None:
                BOUNDARY_MASK = StaggeredGrid(math.stack([wall_mask_, wall_mask_], channel('vector')), ZERO_GRADIENT, **DOMAIN)
                VEL_BOUNDARY  = BOUNDARY_MASK * wall_vec[wall] * vel
                INLET_NORMAL_MASK = StaggeredGrid(math.stack([wall_maskp_, wall_maskp_], channel('vector')), ZERO_GRADIENT, **DOMAIN) * wall_vec[wall]
            elif wall not in visited_walls:   
                BOUNDARY_MASK+= StaggeredGrid(math.stack([wall_mask_, wall_mask_], channel('vector')), ZERO_GRADIENT, **DOMAIN)
                VEL_BOUNDARY += StaggeredGrid(math.stack([wall_mask_, wall_mask_], channel('vector')), ZERO_GRADIENT, **DOMAIN) * wall_vec[wall] * vel
                INLET_NORMAL_MASK += StaggeredGrid(math.stack([wall_maskp_, wall_maskp_], channel('vector')), ZERO_GRADIENT, **DOMAIN) * wall_vec[wall]
        else:
            OUTLET_MASK = CenteredGrid(wall_maskp_, ZERO_GRADIENT, **DOMAIN)
            OUTLET_NORMAL_MASK = StaggeredGrid(math.stack([wall_maskp_, wall_maskp_], channel('vector')), ZERO_GRADIENT, **DOMAIN) * wall_vec[wall] * -1.0
        visited_walls.append(wall)


        
    for wall_type in wall_types:
        wall_tensor[wall_type] = to_phi_t(wall_tensor[wall_type], dim=2)

    return OBS_WALL_GEOMETRY, wall_tensor, INLET_MASK, OUTLET_MASK, BOUNDARY_MASK, VEL_BOUNDARY, idxs_, INLET_NORMAL_MASK, OUTLET_NORMAL_MASK


def create_geometry3d(params):
    '''
    Takes 3D simulation case parameters as input and generates wall geometry and specifies boundary condition masks
    '''
    
    t = int(params['t']) ## wall thickness
    res = int(params['topopt_domain_res'])
    X = res + 2*t 
    Y = res + 2*t
    Z = res + 2*t
    # start inclusive, end exclusive (end-1 is inclusive)
    start_x, start_y, start_z = t, t, t
    end_x, end_y, end_z = res + t, res + t, res + t
    DOMAIN = dict(x=X, y=Y, z=Z, bounds=Box(x=X, y=Y, z=Z))
    vel_inlet = 1.0
    
    domain_grid = CenteredGrid(0, **DOMAIN)
    domain_grid_staggered = StaggeredGrid((0,0,0) , ZERO_GRADIENT, **DOMAIN)

    D = params['topopt_domain_res']/10

    # Constructing Wall Obstacles
    
    INLET_MASK = None
    OUTLET_MASK = None
    BOUNDARY_MASK = None
    VEL_BOUNDARY = None
    wall_tensor = {}

    wall_tensor['left'] = torch.zeros(t,Y,Z).to(params['device'])
    wall_tensor['right'] = torch.zeros(t,Y,Z).to(params['device'])

    wall_tensor['bottom'] = torch.zeros(X,t,Z).to(params['device'])
    wall_tensor['top'] = torch.zeros(X,t,Z).to(params['device'])

    wall_tensor['back'] = torch.zeros(X,Y,t).to(params['device'])
    wall_tensor['front'] = torch.zeros(X,Y,t).to(params['device'])

    for i,loc in enumerate(params['orifice_locs']):
        wall_type = params['orifice_walls'][i]
        
        R = int(D*params['orifice_widths'][i]/2)
        loc_ = [int(loc[0]*D)+t, int(loc[1]*D)+t]
        wall_tensor[wall_type] = construct_orifice3d(wall_tensor[wall_type], loc_, r=R, t=t)  

    wall_types = ['left', 'bottom', 'right', 'top', 'back', 'front']

    
    # ## Adding Wall Obstacles
    num_obstacles = 0
    OBS_WALL_GEOMETRY = []
    obs_coords_list = []
    
    idxs_ = None
    for wall_type in wall_types:
        Z_range = range(0,Z)
        Y_range = range(0,Y)
        X_range = range(0,X)

        offset_x = 0
        offset_y = 0
        offset_z = 0
        if wall_type == 'left':   
            X_range = range(0,start_x)
        elif wall_type == 'right':  
            X_range = range(end_x, X)
            offset_x = end_x
        elif wall_type == 'bottom': 
            Y_range = range(0,start_y)
        elif wall_type == 'top':
            Y_range = range(end_y, Y)
            offset_y = end_y
        elif wall_type == 'back':
            Z_range = range(0, start_z)
        elif wall_type == 'front':
            Z_range = range(end_z, Z)
            offset_z = end_z

        
        idxs = torch.nonzero(wall_tensor[wall_type] == 0)
        
        idxs += torch.tensor([offset_x, offset_y, offset_z]).unsqueeze(0).to(params['device'])
        if idxs_ == None:
            idxs_ = idxs
        else:
            idxs_ = torch.cat([idxs_, idxs], dim=0)

    
        
    obst_grid = torch.zeros(X,Y,Z).to('cuda')

    obst_grid[idxs_[:,0], idxs_[:,1], idxs_[:,2]] = 1
    obst_grid_ = to_phi_t(obst_grid.int())
    OBS_WALL_GEOMETRY = Voxels(UniformGrid(resolution=domain_grid.shape, bounds=Box(x=X, y=Y, z=Z)), obst_grid_)
    

    inlet_walls = []
    for i, vel in enumerate(params['inlet_velocities']):
        if vel!=0:
            if params['orifice_walls'][i] not in inlet_walls:
                inlet_walls.append(params['orifice_walls'][i])
        else:
            outlet_wall = params['orifice_walls'][i]
    
    wall_mask = {}
    wall_maskp = {}
    wall_vec = {}

    wall_mask['left'] = math.concat([to_phi_t(wall_tensor['left'][:1,:,:]),
                                 to_phi_t(torch.zeros(X-1,Y,Z).to(params['device']))], dim=spatial('x'))
    
    wall_mask['right'] =math.concat([to_phi_t(torch.zeros(X-1,Y,Z).to(params['device'])), 
                                  to_phi_t(wall_tensor['right'][:1,:,:])], dim=spatial('x'))
    
    wall_mask['bottom']=math.concat([to_phi_t(wall_tensor['bottom'][:,:1,:]),
                                  to_phi_t(torch.zeros(X,Y-1,Z).to(params['device']))], dim=spatial('y'))
    
    wall_mask['top']   =math.concat([to_phi_t(torch.zeros(X,Y-1,Z).to(params['device']))
                                 ,to_phi_t(wall_tensor['top'][:,:1,:])], dim=spatial('y'))
    
    wall_mask['back']  =math.concat([to_phi_t(wall_tensor['back'][:,:,:1]),
                                to_phi_t(torch.zeros(X,Y,Z-1).to(params['device']))],dim=spatial('z'))
    
    wall_mask['front'] =math.concat([to_phi_t(torch.zeros(X,Y,Z-1).to(params['device'])),
                                to_phi_t(wall_tensor['front'][:,:,:1])], dim=spatial('z'))
    wall_vec['left'] = (1,0,0)
    wall_vec['right']= (-1,0,0)
    wall_vec['bottom']=(0,1,0)
    wall_vec['top']   =(0,-1,0)
    wall_vec['back']  =(0,0,1)
    wall_vec['front'] =(0,0,-1)

    wall_maskp['left'] = math.concat([to_phi_t(torch.zeros(t-1,Y,Z).to(params['device'])), to_phi_t(wall_tensor['left'][:1,:,:].to(params['device'])),
                              to_phi_t(torch.zeros(X-t,Y,Z).to(params['device']))], dim=spatial('x'))

    wall_maskp['right'] =math.concat([to_phi_t(torch.zeros(X-t,Y,Z).to(params['device'])), 
                               to_phi_t(wall_tensor['right'][:1,:,:].to(params['device'])),to_phi_t(torch.zeros(t-1,Y,Z).to(params['device'])) ], dim=spatial('x'))

    wall_maskp['bottom']=math.concat([to_phi_t(torch.zeros(X, t-1,Z).to(params['device'])),  to_phi_t(wall_tensor['bottom'][:,:1,:].to(params['device'])),
                               to_phi_t(torch.zeros(X,Y-t,Z).to(params['device']))], dim=spatial('y'))

    wall_maskp['top']   =math.concat([to_phi_t(torch.zeros(X,Y-t,Z).to(params['device']))
                              ,to_phi_t(wall_tensor['top'][:,:1,:].to(params['device'])), to_phi_t(torch.zeros(X, t-1,Z).to(params['device']))], dim=spatial('y'))

    wall_maskp['back']  =math.concat([to_phi_t(torch.zeros(X, Y, t-1).to(params['device'])), to_phi_t(wall_tensor['back'][:,:,:1].to(params['device'])),
                             to_phi_t(torch.zeros(X,Y,Z-t).to(params['device']))],dim=spatial('z'))

    wall_maskp['front'] =math.concat([to_phi_t(torch.zeros(X,Y,Z-t).to(params['device'])),
                                to_phi_t(wall_tensor['front'][:,:,:1].to(params['device'])), to_phi_t(torch.zeros(X, Y, t-1).to(params['device']))], dim=spatial('z'))


    for i, vel in enumerate(params['inlet_velocities']):
        wall = params['orifice_walls'][i]
        wall_mask_ = wall_mask[wall]
        wall_maskp_ = wall_maskp[wall]
        if abs(vel)>0:
            if INLET_MASK == None:
                INLET_MASK = CenteredGrid(wall_maskp_, ZERO_GRADIENT, **DOMAIN)
            else:
                INLET_MASK += CenteredGrid(wall_maskp_, ZERO_GRADIENT, **DOMAIN)
            if VEL_BOUNDARY==None:
                BOUNDARY_MASK = StaggeredGrid(math.stack([wall_mask_, wall_mask_, wall_mask_], channel('vector')), ZERO_GRADIENT, **DOMAIN)
                VEL_BOUNDARY  = BOUNDARY_MASK * wall_vec[wall] * vel
                INLET_NORMAL_MASK = StaggeredGrid(math.stack([wall_maskp_, wall_maskp_, wall_maskp_], channel('vector')), ZERO_GRADIENT, **DOMAIN) * wall_vec[wall]
            else:   
                BOUNDARY_MASK+= StaggeredGrid(math.stack([wall_mask_, wall_mask_, wall_mask_], channel('vector')), ZERO_GRADIENT, **DOMAIN)
                VEL_BOUNDARY += StaggeredGrid(math.stack([wall_mask_, wall_mask_, wall_mask_], channel('vector')), ZERO_GRADIENT, **DOMAIN) * wall_vec[wall] * vel
                INLET_NORMAL_MASK += StaggeredGrid(math.stack([wall_maskp_, wall_maskp_, wall_maskp_], channel('vector')), ZERO_GRADIENT, **DOMAIN) * wall_vec[wall]
            
        else:
            OUTLET_MASK = CenteredGrid(wall_maskp_, ZERO_GRADIENT, **DOMAIN)
            OUTLET_NORMAL_MASK = StaggeredGrid(math.stack([wall_maskp_, wall_maskp_, wall_maskp_], channel('vector')), ZERO_GRADIENT, **DOMAIN) * wall_vec[wall] * -1.0


        
    for wall_type in wall_types:
        wall_tensor[wall_type] = to_phi_t(wall_tensor[wall_type])

    return OBS_WALL_GEOMETRY, wall_tensor, INLET_MASK, OUTLET_MASK, BOUNDARY_MASK, VEL_BOUNDARY, idxs_, INLET_NORMAL_MASK, OUTLET_NORMAL_MASK




