import cv2
import sys
import os, shutil

def detect(path="images_org/", move_to="images_face", cascade_file = "./lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    if not os.path.exists(path):
        raise RuntimeError("%s: not found" % path)

    if not os.path.exists(move_to):
        os.mkdir(move_to)

    cascade = cv2.CascadeClassifier(cascade_file)
    images_path = []

    for f in os.listdir(path):
        images_path.append(os.path.join(path, f))

    for i,filename in enumerate(images_path):
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(gray,
                                            # detector options
                                            scaleFactor = 1.1,
                                            minNeighbors = 5,
                                            minSize = (128, 128))
        print(filename,len(faces))
        for j, (x, y, w, h) in enumerate(faces):
            cv2.imwrite(os.path.join(move_to,f"chino{i:4d}_{j:2d}.png"), image[y:y+h, x:x+w])


if __name__ == "__main__":
    detect()
