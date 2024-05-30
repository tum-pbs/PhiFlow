#from phi.torch.flow import *

import numpy as np
from PIL import Image as im
import sys
import k3d
import meshplot as mp
from skimage import measure
from phi.torch.flow import *

np.set_printoptions(threshold=sys.maxsize)

obs_list = []





def write_vtk(u_pot,v_pot, u, v ,i, cell_locs,phi, only_geometry=False, additional_args=None):
    f = open(f'potential_flow{i}.vtk', mode="w")
    
    new_line = "\n"
    f.write("# vtk DataFile Version 2.0\n")
    f.write("Potential Flow Data\n")
    f.write("ASCII\n")
    f.write("DATASET STRUCTURED_GRID\n")
    if additional_args!=None:
        X, Y, start_x, start_y, end_x, end_y, obs_list = additional_args
        cell_locs = assign_obstacles(X, Y, start_x, start_y, end_x, end_y, obs_list)

    f.write(f"DIMENSIONS {u.shape[0]+1} {u.shape[1]+1} 1{new_line}")
    f.write(f"POINTS {(u.shape[0]+1)*(u.shape[1]+1)} float{new_line}")

    for j in range(u.shape[0]+1):
        for i in range(u.shape[1]+1):
            f.write(f"{i} {j} 0{new_line}")
    
    f.write(f"CELL_DATA {(u.shape[0])*(u.shape[1])}{new_line}")

    f.write("VECTORS actual_vel float\n")

    for j in range(u.shape[0]):
        for i in range(u.shape[1]):
            f.write(f"{u[i,j]} {v[i,j]} {0.0} {new_line}")

    if only_geometry == False:
        f.write("VECTORS potential_vel float\n")
        
        for j in range(u_pot.shape[0]):
            for i in range(u_pot.shape[1]):
                f.write(f"{u_pot[i,j]} {v_pot[i,j]} {0.0} {new_line}")

        f.write("SCALARS phi float\n")
        f.write("LOOKUP_TABLE default\n")
        
        for j in range(phi.shape[0]):
            for i in range(phi.shape[1]):
                f.write(f"{phi[i,j]}{new_line}")

        

    if len(cell_locs) >0:
        f.write("SCALARS cell_locs int\n")
        f.write("LOOKUP_TABLE default\n")
        
        for j in range(cell_locs.shape[0]):
            for i in range(cell_locs.shape[1]):
                f.write(f"{int(cell_locs[i,j])}{new_line}")


def write_vtk_darcy3d(u, v ,w, index, cell_locs, additional_args=None, params=None):
    f = open(f'{params["sim_name"]}_{index}.vtk', mode="w")
    
    new_line = "\n"
    f.write("# vtk DataFile Version 2.0\n")
    f.write("TopOpt Flow Data\n")
    f.write("ASCII\n")
    f.write("DATASET STRUCTURED_GRID\n")

    f.write(f"DIMENSIONS {u.shape[0]+1} {u.shape[1]+1} {u.shape[2]+1}{new_line}")
    f.write(f"POINTS {(u.shape[0]+1)*(u.shape[1]+1)*(u.shape[2]+1)} float{new_line}")

    for k in range(u.shape[2]+1):
        for j in range(u.shape[1]+1):
            for i in range(u.shape[0]+1):
                f.write(f"{i} {j} {k}{new_line}")

    f.write(f"CELL_DATA {(u.shape[0])*(u.shape[1])*(u.shape[2])}{new_line}")

    f.write("VECTORS actual_vel float\n")

    for k in range(u.shape[2]):
        for j in range(u.shape[1]):
            for i in range(u.shape[0]):
                f.write(f"{u[i,j,k]} {v[i,j,k]} {w[i,j,k]}{new_line}")

    if additional_args!=None:
        X, Y,Z, start_x, start_y, start_z, end_x, end_y, end_z, obs_list, darcy_params = additional_args

        f.write("SCALARS darcy_params float\n")
        f.write("LOOKUP_TABLE default\n")
    
        for k in range(darcy_params.shape[2]):
            for j in range(darcy_params.shape[1]):
                for i in range(darcy_params.shape[0]):
                    f.write(f"{darcy_params[i,j,k]}{new_line}")

    f.write(f"{params}")
    f.close()


def write_vtk_darcy(u_pot,v_pot, u, v ,index, cell_locs,phi, only_geometry=False, additional_args=None, params=None):
    f = open(f'{params["sim_name"]}_{index}.vtk', mode="w")
    
    new_line = "\n"
    f.write("# vtk DataFile Version 2.0\n")
    f.write("TopOpt Flow Data\n")
    f.write("ASCII\n")
    f.write("DATASET STRUCTURED_GRID\n")

    f.write(f"DIMENSIONS {u.shape[0]+1} {u.shape[1]+1} 1{new_line}")
    f.write(f"POINTS {(u.shape[0]+1)*(u.shape[1]+1)} float{new_line}")

    for j in range(u.shape[0]+1):
        for i in range(u.shape[1]+1):
            f.write(f"{i} {j} 0{new_line}")

    f.write(f"CELL_DATA {(u.shape[0])*(u.shape[1])}{new_line}")

    f.write("VECTORS actual_vel float\n")

    for j in range(u.shape[0]):
        for i in range(u.shape[1]):
            f.write(f"{u[i,j]} {v[i,j]} {0.0} {new_line}")

    if additional_args!=None:
        X, Y, start_x, start_y, end_x, end_y, obs_list, darcy_params = additional_args

        f.write("SCALARS darcy_params float\n")
        f.write("LOOKUP_TABLE default\n")
    
        for j in range(darcy_params.shape[0]):
            for i in range(darcy_params.shape[1]):
                f.write(f"{darcy_params[i,j]}{new_line}")
        
        data = im.fromarray(darcy_params.cpu().detach().numpy().transpose((1,0))[::-1,:].astype('uint8')*255)
        

    if only_geometry == False:
        f.write("VECTORS potential_vel float\n")
        
        for j in range(u_pot.shape[0]):
            for i in range(u_pot.shape[1]):
                f.write(f"{u_pot[i,j]} {v_pot[i,j]} {0.0} {new_line}")

        f.write("SCALARS phi float\n")
        f.write("LOOKUP_TABLE default\n")
        
        for j in range(phi.shape[0]):
            for i in range(phi.shape[1]):
                f.write(f"{phi[i,j]}{new_line}")


    
    if params == None:
        data.save(f'TopOpt{index}.png')
    else:
        f.write(f"{params}")
        data.save(f'TopOpt{index}_{params["sim_name"]}.png')

    f.close()


def read_vtk(filename):

    f = open(filename)
    content = f.readlines()

    num_points_data = content[4][11:]
    num_points_x = int(num_points_data.split()[0])
    num_points_y = int(num_points_data.split()[1]) 
    num_points = num_points_x * num_points_y
    print(num_points_x * num_points_y)
    v_data = np.loadtxt(filename, usecols=[0,1], skiprows=num_points+8, max_rows=(num_points_x-1)*(num_points_y-1), dtype=float)
    obstacle_data = np.loadtxt(filename, usecols=0, skiprows=num_points+(num_points_x-1)*(num_points_y-1)+10, 
                        max_rows=(num_points_x-1)*(num_points_y-1), dtype=int)


    v_data = v_data.reshape((num_points_y-1,num_points_x-1,-1)).transpose(1,0,2)
    v_data_ = v_data.transpose(2,1,0)[:,::-1,:] 
    #v_data_ : (2, y, x) i.e y coordinate is row-wise and x coordinate is columnwise in both velx: [0,:,;] and vely: [1,:,:]
    v_staggered_x = np.zeros((num_points_x-1, num_points_x-2))
    v_staggered_y = np.zeros((num_points_x-2, num_points_y-1))
    for i in range(v_data.shape[1]-1):
        v_staggered_x[:, i] = (v_data[:,i,0] + v_data[:,i+1,0])/2.0

    for i in range(v_data.shape[0]-1):
        v_staggered_y[i,:] = (v_data[i,:,1] + v_data[i+1,:,1])/2.0

    v_staggered = np.concatenate([v_staggered_x[:-1, :, None], v_staggered_y[:,:-1, None]], axis=-1)
    
    v_staggered = np.concatenate([np.zeros((v_staggered.shape[0], 1, 2)), v_staggered, np.zeros((v_staggered.shape[0], 1, 2))], axis=1)
    v_staggered = np.concatenate([np.zeros((1, v_staggered.shape[1], 2)), v_staggered, np.zeros((1, v_staggered.shape[1], 2))], axis=0)
    #print(f'v_staggered shape after padding: {v_staggered.shape}')

    obstacle_data = obstacle_data.reshape((num_points_y-1, num_points_x-1)).transpose(1,0)#[::-1,:]

    dict_string = content[num_points+ 2*(num_points_x-1)*(num_points_y-1)+10].replace("'",'"')
    params = eval(dict_string)

    v_tensor = math.tensor(v_staggered.transpose((1,0,2)), spatial('x,y'), channel(vector='x,y'))

    X = 1.2 * params['topopt_domain_res'] 
    Y = 1.2 * params['topopt_domain_res']
    DOMAIN = dict(x=X, y=Y, bounds=Box(x=X, y=Y))


    v = StaggeredGrid(v_tensor, ZERO_GRADIENT, **DOMAIN)

    return v, obstacle_data, params

def plot_voxels(data, color=0x3f6bc5):

    plot = k3d.plot()
    figs_per_line = data.shape[0]/2
    
    for i in range(data.shape[0]):
        trans_vec = np.array([(i%figs_per_line) - figs_per_line/2 + 0.5, 0, (i//figs_per_line)-0.5])
        plot+= k3d.voxels(data[i,0,...].cpu().detach().numpy(), color_map=color, 
                          outlines=False, bounds=[-0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
                          translation=trans_vec*2)
        
    return plot

def plot_points(data, color=0x3f6bc5):

    plot = k3d.plot()
    figs_per_line = data.shape[0]/2
    
    for i in range(data.shape[0]):

        occ_grid = data[i,0,...].cpu().detach().numpy()
        points = np.argwhere(occ_grid)/occ_grid.shape[0] - 0.5
        trans_vec = np.array([(i%figs_per_line) - figs_per_line/2 + 0.5, (i//figs_per_line)-0.5, 0])
        points += trans_vec * 1.2
        plot += k3d.points(points, point_size=1.5/occ_grid.shape[0], shader='3d', color=color)

    return plot


def write_stl_face(fname, face_name, vertices, faces):
    f = open(fname, 'a')

    f.write(f'solid {fname}_{face_name} \n')
    for (v1, v2, v3) in faces:
        v1,v2,v3 = vertices[v1], vertices[v2], vertices[v3]
        vec1 = v2 - v1
        vec2 = v3 - v1
        n = np.cross(vec1, vec2)

        f.write(f'facet normal {n[0]} {n[1]} {n[2]} \n')
        f.write(f'  outer loop\n')
        f.write(f'      vertex {v1[0]} {v1[1]} {v1[2]}\n')
        f.write(f'      vertex {v2[0]} {v2[1]} {v2[2]}\n')
        f.write(f'      vertex {v3[0]} {v3[1]} {v3[2]}\n')
        f.write(f'  endloop\n')
        f.write(f'endfacet\n')

    f.write(f'endsolid {fname}_{face_name}\n')

    f.close()

def write_stl_geometry(fname, vertices, faces, inlet_face_names):
    
    domain_faces = []
    
    inlet_faces = {}
    inlet_face_locs = {}

    outlet_faces = []
    
    
    x_min, x_max = 100,0
    y_min, y_max = 100,0
    z_min, z_max = 100,0

    for vertex in vertices:
        if vertex[0] < x_min: x_min = vertex[0]
        if vertex[0] > x_max: x_max = vertex[0]
        if vertex[1] < y_min: y_min = vertex[1]
        if vertex[1] > y_max: y_max = vertex[1]
        if vertex[2] < z_min: z_min = vertex[2]
        if vertex[2] > z_max: z_max = vertex[2]
    

    print('Domain extremes:')
    print(f'x_min(left): {x_min}, x_max(right): {x_max}, y_min(bottom): {y_min}, y_max(top): {y_max}, z_min(back): {z_min}, z_max(front): {z_max}')

    inlet_face_locs['left'] = x_min
    inlet_face_locs['bottom'] = y_min
    inlet_face_locs['top'] = y_max
    inlet_face_locs['back'] = z_min
    inlet_face_locs['front'] = z_max

    for name in inlet_face_names:
        if name != 'right': 
            inlet_faces[name] = [] 

    for (v1,v2,v3) in faces:
        flag=0
        for name in inlet_face_names:
            if name=='left': idx=0 #right face is always the outlet here
            elif name=='bottom' or name=='top': idx=1
            else: idx=2

            ## If all 3 vertices of triangle element have one common extreme coordinate value, they must be at an extreme face
            if vertices[v1][idx] == inlet_face_locs[name] and vertices[v2][idx] == inlet_face_locs[name] and vertices[v3][idx] == inlet_face_locs[name]:
                inlet_faces[name].append([v1,v2,v3])
                flag=1
                break
        if flag==1: continue
        if vertices[v1][0] == x_max and vertices[v2][0] == x_max and vertices[v3][0] == x_max:
            outlet_faces.append([v1,v2,v3])
        else: domain_faces.append([v1,v2,v3])
    
    write_stl_face(fname, 'domain', vertices, domain_faces)
    for name in inlet_face_names:
        write_stl_face(fname, f'inlet_{name}', vertices, inlet_faces[name])
    write_stl_face(fname, 'outlet_right', vertices, outlet_faces)


def smooth_laplace(vertices, faces, res, num_steps=2):
    """To smoothen predicted mesh for better flow simulation in CAD tools"""

    for step in range(num_steps):
        ## Laplacian mesh smoothing
        vertex_dict_list = []
        for vertex in vertices:
            vertex_dict = {}
            vertex_dict['x'] = vertex[0]
            vertex_dict['y'] = vertex[1]
            vertex_dict['z'] = vertex[2]
            vertex_dict['new_x'] = 0.
            vertex_dict['new_y'] = 0.
            vertex_dict['new_z'] = 0.
            vertex_dict['neighbors'] = []
            vertex_dict_list.append(vertex_dict)

        for (v0, v1, v2) in faces:
            v_tuples = [(v0,v1,v2), (v1,v2,v0), (v2,v0,v1)]
            for v_tuple in v_tuples:
                for v in v_tuple[1:]:
                    if v not in vertex_dict_list[v_tuple[0]]['neighbors']:
                        vertex_dict_list[v_tuple[0]]['new_x'] += vertex_dict_list[v]['x']
                        vertex_dict_list[v_tuple[0]]['new_y'] += vertex_dict_list[v]['y']
                        vertex_dict_list[v_tuple[0]]['new_z'] += vertex_dict_list[v]['z']
                        vertex_dict_list[v_tuple[0]]['neighbors'].append(v)
        smooth_vertices = []
        ## Do not consider smooth coordinates of vertices that were outside the box: [-28,28]
        cap_limit = res//2 - 3
        for i, vertex in enumerate(vertices):
            if abs(vertex_dict_list[i]['x']) > cap_limit or \
               abs(vertex_dict_list[i]['y']) > cap_limit or \
               abs(vertex_dict_list[i]['z']) > cap_limit:
               vertex_dict_list[i]['new_x'] = vertex_dict_list[i]['x'] * len(vertex_dict_list[i]['neighbors'])
               vertex_dict_list[i]['new_y'] = vertex_dict_list[i]['y'] * len(vertex_dict_list[i]['neighbors'])
               vertex_dict_list[i]['new_z'] = vertex_dict_list[i]['z'] * len(vertex_dict_list[i]['neighbors'])
            smooth_vertices.append([vertex_dict_list[i]['new_x']/len(vertex_dict_list[i]['neighbors']), 
                                    vertex_dict_list[i]['new_y']/len(vertex_dict_list[i]['neighbors']),
                                    vertex_dict_list[i]['new_z']/len(vertex_dict_list[i]['neighbors'])]) 
        
        smooth_vertices = np.array(smooth_vertices).astype(np.float32)
        vertices = smooth_vertices


    return vertices



def plot_mesh(name,data, color=0x3f6bc5, sim_name=None):

    #plot = k3d.plot()
    figs_per_line = data.shape[0]/2

    plot_yet=0
    for i in range(data.shape[0]):
        
        if data.shape[0] > 1:
            trans_vec = np.array([(i%figs_per_line) - figs_per_line/2 + 0.5, (i//figs_per_line)-0.5, 0])
        else: trans_vec = np.array([0,0,0])

        occ_grid = data[i,0,...].cpu().detach().numpy()
        occ_grid = occ_grid * 2 - 1 ##[0 to 1] -> [-1 to 1]

        ## Append additional grid cells to get rid of free edges
        L = occ_grid.shape[0]
        occ_grid = np.concatenate([-1.0 * np.ones((1,L,L)) , occ_grid, -1.0 * np.ones((1,L,L))], axis=0)
        occ_grid = np.concatenate([-1.0 * np.ones((L+2,1,L)) , occ_grid, -1.0 * np.ones((L+2,1,L))], axis=1)
        occ_grid = np.concatenate([-1.0 * np.ones((L+2,L+2,1)) , occ_grid, -1.0 * np.ones((L+2,L+2,1))], axis=2)

        vertices, faces, *_ = measure.marching_cubes(occ_grid, level=0)

        
        ## Shift all vertices to locations around center
        vertices -= np.array([1, 1, 1]) * data.shape[-1]/2
        ## Smoothen the mesh 
        smooth_vertices = smooth_laplace(vertices, faces, res=data.shape[-1], num_steps=10)
        
        inlet_face_names = []
        orifice_list = sim_name[0].split('[')[1].split(']')[0].split('_')
        if int(orifice_list[0]) > 0: inlet_face_names.append('left')
        if int(orifice_list[1]) > 0: inlet_face_names.append('bottom')
        if int(orifice_list[3]) > 0: inlet_face_names.append('top')
        if int(orifice_list[4]) > 0: inlet_face_names.append('back')
        if int(orifice_list[5]) > 0: inlet_face_names.append('front')
        print(f'inlet_face_names: {inlet_face_names}')
        write_stl_geometry(f'manifold_mesh_{name}.stl', smooth_vertices, faces, inlet_face_names)
        
        if plot_yet==0:
            plot_smooth = mp.plot(smooth_vertices + 50*trans_vec*1.8, faces, c=color,return_plot=True)
            plot_simple = mp.plot(vertices + 50*trans_vec*1.8, faces, c=color, return_plot=True)
            plot_yet = 1
        else:
            plot_smooth.add_mesh(smooth_vertices + 50*trans_vec*1.8, faces, c=color)
            plot_simple.add_mesh(vertices + 50*trans_vec*1.8, faces, c=color)
    
    return plot_smooth, plot_simple
        

def plot_batch(data, color=0x3f6bc5, representation='points', name=None, sim_name=None):
    
    data = (data + 1)/2 
    data = (data>0.5).float() 
    ## Expand in x,y,z by 3 cells each for better inlet/outlet visualization
    data = torch.cat([data[:,:,0,:,:].unsqueeze(2),  data[:,:,0,:,:].unsqueeze(2),  data[:,:,0,:,:].unsqueeze(2),
                data, data[:,:,-1,:,:].unsqueeze(2), data[:,:,-1,:,:].unsqueeze(2), data[:,:,-1,:,:].unsqueeze(2)], dim=2)
    
    data = torch.cat([ data[:,:,:,0,:].unsqueeze(3),  data[:,:,:,0,:].unsqueeze(3),  data[:,:,:,0,:].unsqueeze(3),
                data, data[:,:,:,-1,:].unsqueeze(3), data[:,:,:,-1,:].unsqueeze(3), data[:,:,:,-1,:].unsqueeze(3)], dim=3)
    
    data = torch.cat([ data[:,:,:,:,0].unsqueeze(4), data[:,:,:,:,0].unsqueeze(4),  data[:,:,:,:,0].unsqueeze(4),
                data, data[:,:,:,:,-1].unsqueeze(4), data[:,:,:,:,-1].unsqueeze(4), data[:,:,:,:,-1].unsqueeze(4)], dim=4)
    
    if representation=='points': plot = plot_points(data, color)
    elif representation=='mesh': 
        plot_smooth, plot_simple = plot_mesh(name,data,color, sim_name)
        #if name.startswith('Pred_mesh'):
        plot_smooth.save(f'plot_smooth_{name}.html')
        plot_simple.save(f'plot_{name}.html')
        #    exit(0)
        return

    else: plot = plot_voxels(data, color)

    plot.screenshot_scale = 4.0
    plot.grid_visible = False
    plot.camera_auto_fit = False
    plot.camera_fov = 60.0  
    
    
    
    with open(f'{name}.html', 'w') as fp:
        
        fp.write(plot.get_snapshot())

    plot.display()

## Potential Flow functions----------------------
def fluid_neighbors(i,j, cell_locs):
    neighbors = []
    if i+1 < cell_locs.shape[0] and cell_locs[i+1,j] == 0:
        neighbors.append((i+1,j))
    if j+1 < cell_locs.shape[1] and cell_locs[i,j+1] == 0:
        neighbors.append((i,j+1))
    if i-1 > 0 and cell_locs[i-1,j] == 0:
        neighbors.append((i-1,j))
    if j-1 > 0 and cell_locs[i,j-1] == 0:
        neighbors.append((i,j-1))

    return neighbors


def assign_obstacles(X, Y, start_x, start_y, end_x, end_y, obs_list):

    cell_locs = np.zeros((X,Y))

    cell_locs[:start_x, :] = 1
    cell_locs[end_x:,:] = 1
    cell_locs[:,:start_y] = 1
    cell_locs[:,end_y:] = 1

    for OBSTACLE in obs_list:
        cell_locs[OBSTACLE[0], OBSTACLE[1]] = 1

    # Inlet 
    cell_locs[:start_x, int(9*start_y): int(10*start_y)] = -1


    # Outlet
    # cell_locs[int(7*start_x): int(8*start_x), :start_y] = -1
    # cell_locs[start_x:end_x, :start_y] = -1
    # cell_locs[end_x:, int(10*Y/64):int(15*Y/64)] = -1
    
    # S-bend
    # cell_locs[end_x:, int(2*start_y):int(3*start_y)] = -1
    
    # 90-bend
    cell_locs[int(9*start_x): int(10*start_x), :start_y] = -1

    return cell_locs


def u_potential(u_actual,v_actual,X,Y,start_x, start_y, end_x, end_y, OBSTACLES=[],k=0):

    dx = 1
    dy = 1

    # start_x, start_y = int(2*X/64), int(2*Y/64) 
    # end_x, end_y = int(62*X/64), int(62*Y/64)

    phi = 100*np.ones((X,Y))

    ## Inlet Fluid Cells (cell_locs[inlet]=-1)

    
    # phi[:start_x, int(25*start_y): int(27.5*start_y)] = 5.0
    
    
    phi[:start_x, int(9*start_y): int(10*start_y)] = 5.0
    # phi[:start_x, start_y: end_y] = 5.0

    

    ## Outlet Fluid Cells (cell_locs[outlet]=-1) #1

    
    # phi[end_x:, int(10*Y/64):int(15*Y/64)] = 15.0
    phi[end_x:, int(2*start_y):int(3*start_y)] = 195.0

    # Outlet Fluid Cells (cell_locs[outlet]=-1) #2

    
    # phi[int(7*start_x): int(8*start_x), :start_y] = 15
    # phi[start_x:end_x, :start_y] = 15

    u = np.zeros_like(phi)
    v = np.zeros_like(phi)

    cell_locs = assign_obstacles(X, Y, start_x, start_y, end_x, end_y, OBSTACLES)
    
    # initializing u_prev, v_prev to u,v respectively makes them their aliases? 
    u_prev = 0
    v_prev = 0

    norm = 10
    iter = 0
    while(norm > 1e-5):

        # Computing phi values via laplace equation
        for i in range(cell_locs.shape[0]):
            for j in range(cell_locs.shape[1]):
                if cell_locs[i,j] == 0:
                    phi[i,j] = phi[i,j+1] + phi[i,j-1]
                    phi[i,j] += phi[i-1,j] + phi[i+1,j]
                    phi[i,j] /= 4

        # Assigning boundary conditions
        for i in range(cell_locs.shape[0]):
            for j in range(cell_locs.shape[1]):
                if cell_locs[i,j] == 1:
                    f_neighbors = fluid_neighbors(i,j,cell_locs)
                    ## dphi/dn = 0 at obstacles
                    phi[i,j] = 0
                    for f_neighbor in f_neighbors:
                        phi[i,j] += phi[f_neighbor[0], f_neighbor[1]]
                    if len(f_neighbors)>0:
                        phi[i,j] /= len(f_neighbors)
        
        # Compute the velocity field grad(phi)
        for i in range(cell_locs.shape[0]):
            for j in range(cell_locs.shape[1]):
                if cell_locs[i,j] == 0:
                    u[i,j] = (phi[i+1,j] - phi[i,j])/dx
                    v[i,j] = (phi[i,j+1] - phi[i,j])/dy

        norm = np.sum( (u-u_prev)**2 + (v-v_prev)**2)
        # print(f'Iter: {iter} Norm: {norm}')
        u_prev = np.copy(u)
        v_prev = np.copy(v)

        # Interpolating velocity at cell centers
        u_ = np.zeros_like(u)
        v_ = np.zeros_like(v)
        for i in range(cell_locs.shape[0]):
            for j in range(cell_locs.shape[1]):
                if cell_locs[i,j] == 0:
                    u_[i,j] = (u[i,j] + u[i-1,j])/2 
                    v_[i,j] = (v[i,j] + v[i,j-1])/2
        iter += 1
    vel = np.concatenate([u_[..., None],v_[..., None]], axis=-1)
    vel = vel[start_x:end_x, start_y:end_y,:]
    print('Potential Flow Solution Succesfully Computed')
    
    return vel

#----------------------

# def read_vtk3d(filename):

#     f = open(filename)
#     content = f.readlines()

#     num_points_data = content[4][11:]
#     num_points_x = int(num_points_data.split()[0])
#     num_points_y = int(num_points_data.split()[1]) 
#     num_points_z = int(num_points_data.split()[2])
#     num_points = num_points_x * num_points_y * num_points_z
#     print(num_points_x * num_points_y * num_points_z)
#     #exit(1)
#     v_data = np.loadtxt(filename, usecols=[0,1,2], skiprows=num_points+8, max_rows=(num_points_x-1)*(num_points_y-1)*(num_points_z-1), dtype=float)
#     obstacle_data = np.loadtxt(filename, usecols=0, skiprows=num_points+(num_points_x-1)*(num_points_y-1)*(num_points_z-1)+10, 
#                         max_rows=(num_points_x-1)*(num_points_y-1)*(num_points_z-1), dtype=int)


#     v_data = v_data.reshape((num_points_z-1,num_points_y-1,num_points_x-1,-1)).transpose(2,1,0,3)
#     v_data_ = v_data.transpose(3,2,1,0)[:,::-1,::-1,:] #v_data_ : (2, y, x) i.e y coordinate is row-wise and x coordinate is columnwise in both velx: [0,:,;] and vely: [1,:,:]
#     #print(v_data_)
#     v_staggered_x = np.zeros((num_points_x-1, num_points_y-2, num_points_z-2))
#     v_staggered_y = np.zeros((num_points_x-2, num_points_y-1, num_points_z-2))
#     v_staggered_z = np.zeros((num_points_x-2, num_points_y-2, num_points_z-1))

#     for i in range(v_data.shape[2]-1):
#         v_staggered_x[:,:,i] = (v_data[:,:,i,0] + v_data[:,:,i+1,0])/2.0

#     for i in range(v_data.shape[1]-1):
#         v_staggered_y[:,i,:] = (v_data[:,i,:,1] + v_data[:,i+1,:,1])/2.0

#     for i in range(v_data.shape[0]-1):
#         v_staggered_z[i,:,:] = (v_data[i,:,:,2] + v_data[i+1,:,:,2])/2.0

#     v_staggered = np.concatenate([v_staggered_x[:-1, :,:, None], v_staggered_y[:,:-1,:, None], v_staggered_z[:,:,:-1,None]], axis=-1)
#     #print(f'v_staggered shape:{v_staggered.shape}')
    
    
#     v_staggered = np.concatenate([np.zeros((v_staggered.shape[0], 1, 1, 3)), v_staggered, np.zeros((v_staggered.shape[0], 1,1, 3))], axis=2)
#     v_staggered = np.concatenate([np.zeros((1, v_staggered.shape[1], 1, 3)), v_staggered, np.zeros((1, v_staggered.shape[1],1, 3))], axis=1)
#     v_staggered = np.concatenate([np.zeros((1, 1, v_staggered.shape[2], 3)), v_staggered, np.zeros((1, 1, v_staggered.shape[2], 3))], axis=0)
#     #print(f'v_staggered shape after padding: {v_staggered.shape}')

#     #exit(1)
#     obstacle_data = obstacle_data.reshape((num_points_z-1, num_points_y-1, num_points_x-1)).transpose(2,1,0)#[::-1,:]

#     dict_string = content[num_points+ 2*(num_points_x-1)*(num_points_y-1)+10].replace("'",'"')
#     #v = np.loadtxt(filename, skiprows=num_points+ 2*(num_points_x-1)*(num_points_y-1)+10, dtype='str').replace("'",'"')
#     #print(f'v={v}')
#     params = eval(dict_string)
#     #print(params)

#     v_tensor = math.tensor(v_staggered.transpose((1,0,2)), spatial('x,y'), channel(vector='x,y'))

#     X = 1.2 * params['topopt_domain_res'] 
#     Y = 1.2 * params['topopt_domain_res']
#     DOMAIN = dict(x=X, y=Y, bounds=Box(x=X, y=Y))


#     v = StaggeredGrid(v_tensor, ZERO_GRADIENT, **DOMAIN)

#     # OBS = []
#     # for j in range(obstacle_data.shape[0]):
#     #     for i in range(obstacle_data.shape[1]):
#     #         if obstacle_data[i,j] == 0:
#     #             OBS.append(Box['x,y', i:i+1, j:j+1])
#     # print(f'velocity data before reshape:{v_data}')
#     # print(f'obstacle data before reshape:{obstacle_data}')
#     # obstacle_data = obstacle_data.reshape((num_points_y-1, num_points_x-1))
#     # #print(v_data)
#     # print(f'velocity data after reshape:{v_data}')
#     # print(f'obstacle data after reshape:{obstacle_data}')
#     # print(f'np array (5,4):{np.ones((5,4))}')

#     return v, obstacle_data, params
    





                    
                

