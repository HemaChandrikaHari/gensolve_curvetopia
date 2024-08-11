import numpy as np
import matplotlib.pyplot as plt

def read_csv ( csv_path ):
    np_path_XYs = np.genfromtxt ( csv_path , delimiter = ',')
    if np_path_XYs.ndim != 2:
        raise ValueError("The loaded CSV file does not contain the expected 2D data.")
    
    path_XYs = []
    for i in np . unique ( np_path_XYs [: , 0]):
        npXYs = np_path_XYs [ np_path_XYs [: , 0] == i ][: , 1:]
        XYs = []
        for j in np . unique ( npXYs [: , 0]):
            XY = npXYs [ npXYs [: , 0] == j ][: , 1:]
            XYs . append ( XY.reshape(-1,2) )
        path_XYs . append ( np.array(XYs) )   
    return path_XYs

data = read_csv ( r'C:\Users\hmanu\OneDrive\Documents\GenSolve\frag0.csv' )

def plot ( path_XYs,save_path ):
    fig , ax = plt . subplots ( tight_layout = True , figsize =(8 , 8))
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i , XYs in enumerate ( path_XYs ):
        c = colours [ i % len( colours )]
        for XY in XYs :
          ax . plot ( XY [: , 0] , XY [: , 1] , c =c , linewidth =2)
    ax . set_aspect ( 'equal')
    plt.savefig(save_path)
    plt . show ()

plot( data , r'C:\Users\hmanu\OneDrive\Documents\GenSolve\output_image.png')