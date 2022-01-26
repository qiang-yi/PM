# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 08:05:35 2021

@author: qiangy
"""

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy import stats
from KDEpy import FFTKDE


def kde_pyramid(pnt_df,width,kernel,buff,res):

    Xmin,Xmax,Ymin,Ymax=[-buff,width+buff,-buff,width+buff]

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[Xmin:Xmax:(width+1)*1j, Ymin:Ymax:(width+1)*1j]
    # Create the point set to create kernel density
    x,y=np.array(pnt_df['x']),np.array(pnt_df['y'])
    
    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T
    
    # ---- initiate all types of data structures ------------
    dim_x,dim_y,dim_z = xx.shape[0],yy.shape[0],int((xx.shape[0] + 1)/2)
    
    array3D_raw=np.empty((dim_y, dim_x, dim_z))
    array3D_norm=np.empty((dim_y, dim_x, dim_z))
    array3D_z=np.empty((dim_y, dim_x, dim_z))
    array3D_pct=np.empty((dim_y, dim_x, dim_z))
    
    # The ratio of shrinking the grid, if 10, the coordinates divided by 10
    num_grid=1
    
    # mesh3D_norm=np.empty((xx.shape[0]*xx.shape[1],4))
    # mesh3D_norm[:,0] = ((xx-np.min(xx))/num_grid).astype(int).flatten()
    # mesh3D_norm[:,1] = ((yy-np.min(yy))/num_grid).astype(int).flatten()
    # mesh3D_norm[:,2] = 0
    
    # mesh3D_z = np.empty((xx.shape[0]*xx.shape[1],4))
    # mesh3D_z[:,0] = ((xx-np.min(xx))/num_grid).astype(int).flatten()
    # mesh3D_z[:,1] = ((yy-np.min(yy))/num_grid).astype(int).flatten()
    # mesh3D_z[:,2] = 0
    
    # mesh3D_pct = np.empty((xx.shape[0]*xx.shape[1],4))
    # mesh3D_pct[:,0] = ((xx-np.min(xx))/num_grid).astype(int).flatten()
    # mesh3D_pct[:,1] = ((yy-np.min(yy))/num_grid).astype(int).flatten()
    # mesh3D_pct[:,2] = 0
    
    if len(res)<(dim_z-1):
        print('Length of z-scale is smaller than dim_z')
        exit()

    
    # ---------- Compute kernel density at different scales --------
    for i in range(0,len(res)):
    #for bw in res:
        bw = res[i]
        #for bw in bw_ls:
    
        kde_skl = KernelDensity(bandwidth=bw,kernel=kernel).fit(xy_train)
        # score_samples() returns the log-likelihood of the samples
        value = np.exp(kde_skl.score_samples(xy_sample))
        value = np.reshape(value,xx.shape)
        
        
        
        x1 = ((xx-np.min(xx))/num_grid).astype(int).flatten()
        y1 = ((yy-np.min(yy))/num_grid).astype(int).flatten()
        z1 = np.zeros(x1.shape[0]) + i
        #print("max z1:"+str(np.mean(z1)))
        #print("z1.shape v1:"+str(x1.shape[0]))
        v1 = value.flatten()
        
        # rescale to [0,1]
        v1_norm=(v1 - np.nanmin(v1))/(np.nanmax(v1)-np.nanmin(v1))
        
        # rescale to z-score
        v1_z=stats.zscore(v1)
        
        # rescale to percentile scores [0 - 1]
        v1_pct = stats.rankdata(v1, "min")/len(v1)
        
    
        # ----------- create meshgrid of (x, y, z, value) ----------
        # Create 3D arrays of raw value
        z_array_raw = np.nan * np.empty((dim_y, dim_x))
        z_array_raw[y1, x1] = v1
        array3D_raw[:,:,i] = z_array_raw
        
        # Create 3D arrays of norm (0,1)
        z_array_norm = np.nan * np.empty((dim_y, dim_x))
        z_array_norm[y1, x1] = v1_norm
        array3D_norm[:,:,i] = z_array_norm
        
        # Create 3D arrays of z-score
        z_array_z = np.nan * np.empty((dim_y, dim_x))
        z_array_z[y1, x1] = v1_z
        array3D_z[:,:,i] = z_array_z
        
        # Create 3D arrays of z-score percentile
        z_array_pct = np.nan * np.empty((dim_y, dim_x))
        z_array_pct[y1, x1] = v1_pct
        array3D_pct[:,:,i] = z_array_pct
        
        print("layer: "+str(i))
        print("max: "+str(v1_norm.max()))
        print("min: "+str(v1_norm.min()))
        
    return array3D_raw, array3D_norm, array3D_z, array3D_pct
    
def kde_pyramid_KDEpy(pnt_df,width,kernel,buff,res):

    Xmin,Xmax,Ymin,Ymax=[-buff,width+buff,-buff,width+buff]

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[Xmin:Xmax:width*1j, Ymin:Ymax:width*1j]
    # Create the point set to create kernel density
    x,y=np.array(pnt_df['x']),np.array(pnt_df['y'])
    
    
    # ---- initiate all types of data structures ------------
    dim_x,dim_y,dim_z = xx.shape[0],yy.shape[0],int((xx.shape[0] + 1)/2)
    
    array3D_raw=np.empty((dim_y, dim_x, dim_z))
    array3D_norm=np.empty((dim_y, dim_x, dim_z))
    array3D_z=np.empty((dim_y, dim_x, dim_z))
    array3D_pct=np.empty((dim_y, dim_x, dim_z))
    
    # The ratio of shrinking the grid, if 10, the coordinates divided by 10
    num_grid=1
    
    if len(res)<(dim_z-1):
        print('Length of z-scale is smaller than dim_z')
        exit()

    
    # ---------- Compute kernel density at different scales --------
    #for i in range(1,dim_z):
    for i in range(0,len(res)):
    #for bw in res:
        bw = res[i] 
        
        #bw = res[i]
        #for bw in bw_ls:
            
        data = np.array(pnt_df)
        grid_points = 500  # Grid points in each dimension
    
        # Compute the kernel density estimate
        kde = FFTKDE(bw= bw, kernel=kernel, norm=2)
        grid, points = kde.fit(data).evaluate(grid_points)
    
        # The grid is of shape (obs, dims), points are of shape (obs, 1)
        x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
        value = points.reshape(grid_points, grid_points).T

        
        x1 = ((xx-np.min(xx))/num_grid).astype(int).flatten()
        y1 = ((yy-np.min(yy))/num_grid).astype(int).flatten()

        #print("max z1:"+str(np.mean(z1)))
        #print("z1.shape v1:"+str(x1.shape[0]))
        v1 = value.flatten()
        
        # rescale to [0,1]
        v1_norm=(v1 - np.nanmin(v1))/(np.nanmax(v1)-np.nanmin(v1))
        
        # rescale to z-score
        v1_z=stats.zscore(v1)
        
        # rescale to percentile scores [0 - 1]
        v1_pct = stats.rankdata(v1, "min")/len(v1)
        
    
        # ----------- create meshgrid of (x, y, z, value) ----------
        # Create 3D arrays of raw value
        z_array_raw = np.nan * np.empty((dim_y, dim_x))
        z_array_raw[y1, x1] = v1
        array3D_raw[:,:,i] = z_array_raw
        
        # Create 3D arrays of norm (0,1)
        z_array_norm = np.nan * np.empty((dim_y, dim_x))
        z_array_norm[y1, x1] = v1_norm
        array3D_norm[:,:,i] = z_array_norm
        
        # Create 3D arrays of z-score
        z_array_z = np.nan * np.empty((dim_y, dim_x))
        z_array_z[y1, x1] = v1_z
        array3D_z[:,:,i] = z_array_z
        
        # Create 3D arrays of z-score percentile
        z_array_pct = np.nan * np.empty((dim_y, dim_x))
        z_array_pct[y1, x1] = v1_pct
        array3D_pct[:,:,i] = z_array_pct
        
        print("layer: "+str(i))
        print("max: "+str(v1_norm.max()))
        print("min: "+str(v1_norm.min()))
        
    return array3D_raw, array3D_norm, array3D_z, array3D_pct