"""
Copy some functions from tensorlayer
"""
import numpy as np

import tensorflow as tf
import os

def save_npz(save_list=[], name='model.npz', sess=None):
    """Input parameters and the file name, save parameters into .npz file. Use tl.utils.load_npz() to restore.
    Parameters
    ----------
    save_list : a list
        Parameters want to be saved.
    name : a string or None
        The name of the .npz file.
    sess : None or Session
    Examples
    --------
    - Save model to npz
    >>> tl.files.save_npz(network.all_params, name='model.npz', sess=sess)
    - Load model from npz (Method 1)
    >>> load_params = tl.files.load_npz(name='model.npz')
    >>> tl.files.assign_params(sess, load_params, network)
    - Load model from npz (Method 2)
    >>> tl.files.load_and_assign_npz(sess=sess, name='model.npz', network=network)
    Notes
    -----
    If you got session issues, you can change the value.eval() to value.eval(session=sess)
    References
    ----------
    - `Saving dictionary using numpy <http://stackoverflow.com/questions/22315595/saving-dictionary-of-header-information-using-numpy-savez>`_
    """
    ## save params into a list
    save_list_var = []
    if sess:
        save_list_var = sess.run(save_list)
    else:
        try:
            for k, value in enumerate(save_list):
                save_list_var.append(value.eval())
        except:
            print(" Fail to save model, Hint: pass the session into this function, save_npz(network.all_params, name='model.npz', sess=sess)")
    np.savez(name, params=save_list_var)
    save_list_var = None
    del save_list_var
    print("=> %s saved" % name)

    ## save params into a dictionary
    # rename_dict = {}
    # for k, value in enumerate(save_dict):
    #     rename_dict.update({'param'+str(k) : value.eval()})
    # np.savez(name, **rename_dict)
    # print('Model is saved to: %s' % name)

def load_npz(path='', name='model.npz'):
    """Load the parameters of a Model saved by tl.files.save_npz().
    Parameters
    ----------
    path : a string
        Folder path to .npz file.
    name : a string or None
        The name of the .npz file.
    Returns
    --------
    params : list
        A list of parameters in order.
    Examples
    --------
    - See ``save_npz``
    References
    ----------
    - `Saving dictionary using numpy <http://stackoverflow.com/questions/22315595/saving-dictionary-of-header-information-using-numpy-savez>`_
    """
    ## if save_npz save params into a dictionary
    # d = np.load( path+name )
    # params = []
    # print('Load Model')
    # for key, val in sorted( d.items() ):
    #     params.append(val)
    #     print('Loading %s, %s' % (key, str(val.shape)))
    # return params
    ## if save_npz save params into a list
    d = np.load( path+name, encoding='latin1' )
    # for val in sorted( d.items() ):
    #     params = val
    #     return params
    return d['params']
    # print(d.items()[0][1]['params'])
    # exit()
    # return d.items()[0][1]['params']

def assign_params(sess, params, variables):
    """Assign the given parameters to the TensorLayer network.
    Parameters
    ----------
    sess : TensorFlow Session. Automatically run when sess is not None.
    params : a list
        A list of parameters in order.
    network : a :class:`Layer` class
        The network to be assigned
    Returns
    --------
    ops : list
        A list of tf ops in order that assign params. Support sess.run(ops) manually.
    Examples
    --------
    - Save model to npz
    >>> tl.files.save_npz(network.all_params, name='model.npz', sess=sess)
    - Load model from npz (Method 1)
    >>> load_params = tl.files.load_npz(name='model.npz')
    >>> tl.files.assign_params(sess, load_params, network)
    - Load model from npz (Method 2)
    >>> tl.files.load_and_assign_npz(sess=sess, name='model.npz', network=network)
    References
    ----------
    - `Assign value to a TensorFlow variable <http://stackoverflow.com/questions/34220532/how-to-assign-value-to-a-tensorflow-variable>`_
    """
    ops = []
    for idx, param in enumerate(params):
        ops.append(variables[idx].assign(param))
    if sess is not None:
        sess.run(ops)
        print("=> Assignment successful.")
    return ops

def load_and_assign_npz(sess=None, name=None, variables=None):
    """Load model from npz and assign to a network.
    Parameters
    -------------
    sess : TensorFlow Session
    name : string
        Model path.
    network : a :class:`Layer` class
        The network to be assigned
    Returns
    --------
    Returns False if faild to model is not exist.
    Examples
    ---------
    >>> tl.files.save_npz(net.all_params, name='net.npz', sess=sess)
    >>> tl.files.load_and_assign_npz(sess=sess, name='net.npz', network=net)
    """
    assert variables is not None
    assert sess is not None
    if not os.path.exists(name):
        print("=> [!] Load {} failed!".format(name))
        return False
    else:
        params = load_npz(name=name)
        
        import pprint
        vs = []
        ps = []
        MIN = min(len(variables), len(params))
        for i in range(MIN):
            print(variables[i], params[i].shape)
            if variables[i].get_shape() == params[i].shape:
                print("Loaded")
                vs.append(variables[i])
                ps.append(params[i])
        
        assign_params(sess, ps, vs)
        print("=> [*] Load {} SUCCESS!".format(name))
        return variables

## Load and save network dict npz
def save_npz_dict(save_list=[], name='model.npz', sess=None):
    """Input parameters and the file name, save parameters as a dictionary into .npz file.
    Use ``tl.files.load_and_assign_npz_dict()`` to restore.
    Parameters
    ----------
    save_list : a list to tensor for parameters
        Parameters want to be saved.
    name : a string
        The name of the .npz file.
    sess : Session
    """
    assert sess is not None
    save_list_names = [tensor.name for tensor in save_list]
    save_list_var = sess.run(save_list)
    save_var_dict = {save_list_names[idx]: val for idx, val in enumerate(save_list_var)}
    np.savez(name, **save_var_dict)
    save_list_var = None
    save_var_dict = None
    del save_list_var
    del save_var_dict
    print("[*] Model saved in npz_dict %s" % name)

def load_and_assign_npz_dict(name='model.npz', sess=None):
    """Restore the parameters saved by ``tl.files.save_npz_dict()``.
    Parameters
    ----------
    name : a string
        The name of the .npz file.
    sess : Session
    """
    assert sess is not None
    if not os.path.exists(name):
        print("[!] Load {} failed!".format(name))
        return False

    params = np.load(name)
    if len(params.keys()) != len(set(params.keys())):
        raise Exception("Duplication in model npz_dict %s" % name)
    ops = list()
    for key in params.keys():
        try:
            # tensor = tf.get_default_graph().get_tensor_by_name(key)
            # varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=key)
            varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=key)
            if len(varlist) > 1:
                raise Exception("[!] Multiple candidate variables to be assigned for name %s" % key)
            elif len(varlist) == 0:
                raise KeyError
            else:
                ops.append(varlist[0].assign(params[key]))
                print("[*] params restored: %s" % key)
        except KeyError:
            print("[!] Warning: Tensor named %s not found in network." % key)

    sess.run(ops)
    print("[*] Model restored from npz_dict %s" % name)

# def save_npz_dict(save_list=[], name='model.npz', sess=None):
#     """Input parameters and the file name, save parameters as a dictionary into .npz file. Use tl.utils.load_npz_dict() to restore.
#
#     Parameters
#     ----------
#     save_list : a list
#         Parameters want to be saved.
#     name : a string or None
#         The name of the .npz file.
#     sess : None or Session
#
#     Notes
#     -----
#     This function tries to avoid a potential broadcasting error raised by numpy.
#
#     """
#     ## save params into a list
#     save_list_var = []
#     if sess:
#         save_list_var = sess.run(save_list)
#     else:
#         try:
#             for k, value in enumerate(save_list):
#                 save_list_var.append(value.eval())
#         except:
#             print(" Fail to save model, Hint: pass the session into this function, save_npz_dict(network.all_params, name='model.npz', sess=sess)")
#     save_var_dict = {str(idx):val for idx, val in enumerate(save_list_var)}
#     np.savez(name, **save_var_dict)
#     save_list_var = None
#     save_var_dict = None
#     del save_list_var
#     del save_var_dict
#     print("[*] %s saved" % name)
#
# def load_npz_dict(path='', name='model.npz'):
#     """Load the parameters of a Model saved by tl.files.save_npz_dict().
#
#     Parameters
#     ----------
#     path : a string
#         Folder path to .npz file.
#     name : a string or None
#         The name of the .npz file.
#
#     Returns
#     --------
#     params : list
#         A list of parameters in order.
#     """
#     d = np.load( path+name )
#     saved_list_var = [val[1] for val in sorted(d.items(), key=lambda tup: int(tup[0]))]
#     return saved_list_var

## Load and save network ckpt
def save_ckpt(sess=None, mode_name='model.ckpt', save_dir='checkpoint', var_list=[], global_step=None, printable=False):
    """Save parameters into ckpt file.
    Parameters
    ------------
    sess : Session.
    mode_name : string, name of the model, default is ``model.ckpt``.
    save_dir : string, path / file directory to the ckpt, default is ``checkpoint``.
    var_list : list of variables, if not given, save all global variables.
    global_step : int or None, step number.
    printable : bool, if True, print all params info.
    Examples
    ---------
    - see ``tl.files.load_ckpt()``.
    """
    assert sess is not None
    ckpt_file = os.path.join(save_dir, mode_name)
    if var_list == []:
        var_list = tf.global_variables()

    print("[*] save %s n_params: %d" % (ckpt_file, len(var_list)))

    if printable:
        for idx, v in enumerate(var_list):
            print("  param {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))

    saver = tf.train.Saver(var_list)
    saver.save(sess, ckpt_file, global_step=global_step)

def load_ckpt(sess=None, mode_name='model.ckpt', save_dir='checkpoint', var_list=[], is_latest=True, printable=False):
    """Load parameters from ckpt file.
    Parameters
    ------------
    sess : Session.
    mode_name : string, name of the model, default is ``model.ckpt``.
        Note that if ``is_latest`` is True, this function will get the ``mode_name`` automatically.
    save_dir : string, path / file directory to the ckpt, default is ``checkpoint``.
    var_list : list of variables, if not given, save all global variables.
    is_latest : bool, if True, load the latest ckpt, if False, load the ckpt with the name of ```mode_name``.
    printable : bool, if True, print all params info.
    Examples
    ----------
    - Save all global parameters.
    >>> tl.files.save_ckpt(sess=sess, mode_name='model.ckpt', save_dir='model', printable=True)
    - Save specific parameters.
    >>> tl.files.save_ckpt(sess=sess, mode_name='model.ckpt', var_list=net.all_params, save_dir='model', printable=True)
    - Load latest ckpt.
    >>> tl.files.load_ckpt(sess=sess, var_list=net.all_params, save_dir='model', printable=True)
    - Load specific ckpt.
    >>> tl.files.load_ckpt(sess=sess, mode_name='model.ckpt', var_list=net.all_params, save_dir='model', is_latest=False, printable=True)
    """
    assert sess is not None

    if is_latest:
        ckpt_file = tf.train.latest_checkpoint(save_dir)
    else:
        ckpt_file = os.path.join(save_dir, mode_name)

    if var_list == []:
        var_list = tf.global_variables()

    print("[*] load %s n_params: %d" % (ckpt_file, len(var_list)))

    if printable:
        for idx, v in enumerate(var_list):
            print("  param {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))

    try:
        saver = tf.train.Saver(var_list)
        saver.restore(sess, ckpt_file)
    except Exception as e:
        print(e)
        print("[*] load ckpt fail ...")

## Load and save variables
def save_any_to_npy(save_dict={}, name='file.npy'):
    """Save variables to .npy file.
    Examples
    ---------
    >>> tl.files.save_any_to_npy(save_dict={'data': ['a','b']}, name='test.npy')
    >>> data = tl.files.load_npy_to_any(name='test.npy')
    >>> print(data)
    ... {'data': ['a','b']}
    """
    np.save(name, save_dict)

def load_npy_to_any(path='', name='file.npy'):
    """Load .npy file.
    Examples
    ---------
    - see save_any_to_npy()
    """
    file_path = os.path.join(path, name)
    try:
        npy = np.load(file_path).item()
    except:
        npy = np.load(file_path)
    finally:
        try:
            return npy
        except:
            print("[!] Fail to load %s" % file_path)
            exit()

## Folder functions
def file_exists(filepath):
    """ Check whether a file exists by given file path. """
    return os.path.isfile(filepath)

def folder_exists(folderpath):
    """ Check whether a folder exists by given folder path. """
    return os.path.isdir(folderpath)

def del_file(filepath):
    """ Delete a file by given file path. """
    os.remove(filepath)

def del_folder(folderpath):
    """ Delete a folder by given folder path. """
    os.rmdir(folderpath)

def read_file(filepath):
    """ Read a file and return a string.
    Examples
    ---------
    >>> data = tl.files.read_file('data.txt')
    """
    with open(filepath, 'r') as afile:
        return afile.read()

def load_file_list(path=None, regx='\.npz', printable=True):
    """Return a file list in a folder by given a path and regular expression.
    Parameters
    ----------
    path : a string or None
        A folder path.
    regx : a string
        The regx of file name.
    printable : boolean, whether to print the files infomation.
    Examples
    ----------
    >>> file_list = tl.files.load_file_list(path=None, regx='w1pre_[0-9]+\.(npz)')
    """
    if path == False:
        path = os.getcwd()
    file_list = os.listdir(path)
    return_list = []
    for idx, f in enumerate(file_list):
        if re.search(regx, f):
            return_list.append(f)
    # return_list.sort()
    if printable:
        print('Match file list = %s' % return_list)
        print('Number of files = %d' % len(return_list))
    return return_list

def load_folder_list(path=""):
    """Return a folder list in a folder by given a folder path.
    Parameters
    ----------
    path : a string or None
        A folder path.
    """
    return [os.path.join(path,o) for o in os.listdir(path) if os.path.isdir(os.path.join(path,o))]

def exists_or_mkdir(path, verbose=True):
    """Check a folder by given name, if not exist, create the folder and return False,
    if directory exists, return True.
    Parameters
    ----------
    path : a string
        A folder path.
    verbose : boolean
        If True, prints results, deaults is True
    Returns
    --------
    True if folder exist, otherwise, returns False and create the folder
    Examples
    --------
    >>> tl.files.exists_or_mkdir("checkpoints/train")
    """
    if not os.path.exists(path):
        if verbose:
            print("[*] creates %s ..." % path)
        os.makedirs(path)
        return False
    else:
        if verbose:
            print("[!] %s exists ..." % path)
        return True