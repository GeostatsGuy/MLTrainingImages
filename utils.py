import os                       # os operations
import numpy as np              # arrays and operations
import pandas as pd             # dataframes 
from scipy.io import savemat    # save as .mat
from scipy.io import loadmat    # load .mat file
import matplotlib.pyplot as plt # plotting (2D)
import pyvista as pv            # plotting (3D)

class GeoTI:
    def __init__(self, folder):
        self.verbose   = True
        self.save_data = False
        self.save_fig  = False
        self.nx = 256
        self.ny = 256
        self.nz = 128
        self.mydir = folder

    def read_write_data(self):
        self.n_files = len(os.listdir(self.mydir))
        if self.verbose:
            print('Current Directory: {} | Total Files: {}'.format(self.mydir, self.n_files))
        count = 0
        for file in os.listdir(self.mydir):
            count += 1
            if file.endswith('.out'):
                f = open(os.path.join(self.mydir, file))
                self.basename = os.path.splitext(file)[0]
                df = pd.read_csv(f, header=None, skiprows=3)
                
                self.facies = np.reshape(np.flip(np.reshape(np.array(df), 
                                                       [self.nz,self.ny,self.nx]).T), (-1,1))
                self.df_facies = pd.DataFrame(self.facies, columns=['Facies'])
                self.mdic_facies = {"Label":self.basename, "Facies":np.flip(np.array(df))}
                
                if self.save_data:
                    np.save(os.path.join(self.mydir, self.basename+'.npy'), self.facies)
                    self.df_facies.to_csv(os.path.join(self.mydir, self.basename+'.csv'), index=False)
                    savemat(os.path.join(self.mydir, self.basename+'.mat'), self.mdic_facies)
            if self.verbose:
                print('{},'.format(count), end=' ', flush=True)
        if self.verbose:
            print('... DONE!')
    
    def plot_data(self, file_n, slices=[1,16,32,48,64,80,96,112,128], 
                  figsize=(25,3), windowsize=(300,300), jupyterbackend='static',
                  mycmap='viridis'):
        file = os.path.join(self.mydir, os.listdir(self.mydir)[file_n])
        basename = os.path.splitext(file)[0]
        facies = self.facies.reshape(self.nx, self.ny, self.nz)
        fig, axs = plt.subplots(1, len(slices), figsize=figsize, facecolor='white')
        fig.suptitle(os.path.join(self.mydir, file))
        cmap = plt.get_cmap(mycmap, np.max(facies)+1)
        for i in range(len(slices)):
            im = axs[i].imshow(facies[...,slices[i]-1], cmap=cmap)
            axs[i].set(title='Slice {}'.format(slices[i]))
        for k in range(1,len(slices)):
            axs[k].set(xticks=[], yticks=[])
        cb = fig.colorbar(im, ax=axs, pad=0.04, fraction=0.046, location='left',
                          ticks=np.unique(facies), label='facies')
        plt.show()
        if self.save_fig:
            fig.savefig(basename+'_slices'+'.png')
        
        p = pv.Plotter()
        p.add_mesh(np.flip(facies))
        if self.save_fig:
            p.show(window_size=windowsize, 
                   jupyter_backend=jupyterbackend, 
                   screenshot=basename+'_volume'+'.png')
        else:
            p.show(window_size=windowsize, 
                   jupyter_backend=jupyterbackend)
