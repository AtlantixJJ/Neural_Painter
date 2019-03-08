"""
Detect animation faces using OpenCV2 (lbpcascade_animeface.xml).

Auther: Jin Xie, Jianjin Xu.
"""
import cv2
import os.path
from glob import glob
import tqdm

# config
#input_dir = "imgs_safebooru"
#output_dir = "faces_safebooru"#.format(output_size)

input_dir = "imgs_konachan"
output_dir = "faces_konachan"

def detect(filename, cascade_file="lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(96, 96))
    
    for i, (x, y, w, h) in enumerate(faces):
        l = int((w+h)/2)

        face = image[y : y + l, x : x + l, :]
        #face = cv2.resize(face, (output_size, output_size))
        save_filename = '%s-%d.jpg' % (os.path.basename(filename).split('.')[0], i)
        cv2.imwrite(os.path.join(output_dir, save_filename), face)
    return len(faces)


if __name__ == '__main__':
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    file_list = glob('{}/*.jpg'.format(input_dir))
    file_list += glob('{}/*.jpeg'.format(input_dir))
    file_list += glob('{}/*.png'.format(input_dir))
    detected_faces = 0
    for i in tqdm.trange(len(file_list)):
        filename = file_list[i]
        detected_faces += detect(filename)
        print("scanned images: {} ; detected faces: {}".format(i + 1, detected_faces))
