import numpy as np
from lib.analysis import plot, hist

def diff_vector(vecs):
    """
    Compare the difference of two vector
    """
    # change into vector
    for i in range(len(vecs)):
        if len(vecs[i].shape) > 1:
            vecs[i] = vecs[i].reshape(-1)
    
    # diff and hist
    # assumes 2 vector
    hist(vecs[0]-vecs[1], "test/expr/diff_vector.png")
    
def make_square(arr):
    """
    Transform a N*2D array into 2D squares for visualization
    """
    edge_len = int(np.sqrt(arr.shape[0]))
    edge_num = int(np.sqrt(arr.shape[-1]))
    if edge_num ** 2 < arr.shape[-1]:
        edge_num += 1
    arr = arr.reshape(edge_len, edge_len, -1)
    # hard code to 1 channel
    show_image = np.zeros((edge_len * edge_num, edge_len * edge_num))
    for i in range(arr.shape[-1]):
        idx, idy = i // edge_num, i % edge_num
        stx, edx = idx*edge_len, (idx+1)*edge_len
        sty, edy = idy*edge_len, (idy+1)*edge_len
        show_image[stx:edx, sty:edy] = arr[:, :, i]
    
    return show_image


def weight_statistics(var):
    maxx, minn, std = var.max(), var.min(), var.std()

    #print("Max: %f Min: %f Std: %f" % (maxx, minn, std))
    return maxx, minn, std

def visualize_fc(var, print_info=True):
    """
    Args:
    var :   (in_dim, out_dim).
    """
    # Get basic statistics
    if print_info:
        maxx, minn, std = weight_statistics(var)
        print("Max: %f Min: %f Std: %f" % (maxx, minn, std))
        
    # The input will be transformed into a image, placed according to out dim

    
