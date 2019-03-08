import sys
sys.path.insert(0, '.')

import tensorflow as tf
from lib import files

model_dir = "./"
base_name = sys.argv[1]# "goodmodel_dragan_anime1"
model_name = base_name + sys.argv[2]#"-00400000"
model_path = model_dir + model_name
sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph(model_path + ".meta")
saver.restore(sess, model_path)

gen_vars = [v for v in tf.trainable_variables() if v.name.find("Gen") > -1]
disc_vars = [v for v in tf.trainable_variables() if v.name.find("Disc") > -1]
gen_moving_vars = [v for v in tf.all_variables() if v.name.find("Gen") > -1 and v.name.find("moving") > -1]
disc_moving_vars = [v for v in tf.all_variables() if v.name.find("Disc") > -1 and v.name.find("moving") > -1]

files.save_npz(gen_vars + gen_moving_vars, model_dir + base_name + "_gen.npz")
files.save_npz(disc_vars + disc_moving_vars, model_dir + base_name + "_disc.npz")

#files.save_npz(gen_vars + gen_moving_vars, model_dir + "goodmodel_dragan_anime1_gen.npz")
#files.save_npz(disc_vars + disc_moving_vars, model_dir + "goodmodel_dragan_anime1_disc.npz")