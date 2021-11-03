import operator
from skimage.io import imread
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import U2NET
import os
import numpy as np
import glob
from PIL import Image
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage.morphology import binary_erosion
from Video_images import VideoToImages
import shutil
from functools import reduce
import math
import imageio

def naive_cutout(img, mask):
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask.resize(img.size, Image.LANCZOS))
    return cutout

def alpha_matting_cutout(
    img,
    mask,
    foreground_threshold,
    background_threshold,
    erode_structure_size,
    base_size,
):
    size = img.size

    img.thumbnail((base_size, base_size), Image.LANCZOS)
    mask = mask.resize(img.size, Image.LANCZOS)

    img = np.asarray(img)
    mask = np.asarray(mask)

    is_foreground = mask > foreground_threshold
    is_background = mask < background_threshold

    structure = None
    if erode_structure_size > 0:
        structure = np.ones((erode_structure_size, erode_structure_size), dtype=np.int)

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    img_normalized = img / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    cutout = Image.fromarray(cutout)
    cutout = cutout.resize(size, Image.LANCZOS)

    return cutout

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def simplify_contour(contour, n_corners=4):

    n_iter, max_iter = 0, 100
    lb, ub = 0., 1.

    while True:
        n_iter += 1
        if n_iter > max_iter:
            return contour

        k = (lb + ub)/2.
        eps = k*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        if len(approx) > n_corners:
            lb = (lb + ub)/2.
        elif len(approx) < n_corners:
            ub = (lb + ub)/2.
        else:
            return approx

def save_output(image_name,pred,d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR).convert('L')
    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    alpha_matting = True,
    alpha_matting_foreground_threshold = 240,
    alpha_matting_background_threshold = 10,
    alpha_matting_erode_structure_size = 10,
    alpha_matting_base_size = 1000

    img = Image.open(image_name).convert("RGB")
    mask = imo

    if alpha_matting:
        try:
            cutout = alpha_matting_cutout(
                img,
                mask,
                alpha_matting_foreground_threshold,
                alpha_matting_background_threshold,
                alpha_matting_erode_structure_size,
                alpha_matting_base_size,
            )
        except:
            cutout = naive_cutout(img, mask)
    else:
        cutout = naive_cutout(img, mask).con

    cutout_arr = np.asarray(cutout)
    img_gray = cv2.cvtColor(cutout_arr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    _, threshold = cv2.threshold(img_gray, 110, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(threshold, kernel)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours)==1:
        cnt = contours[0]
    else:
        cnt = max(contours, key = cv2.contourArea)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)

    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), box), [len(box)] * 2))
    sorted_arr = (sorted(box, key=lambda coord: (-135 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360))
    sorted_arr = np.asarray(sorted_arr)

    pts = sorted_arr
    pts2conv = [[0, w], [h, w], [h, 0], [0, 0]]

    pts1_arr = np.float32(pts)
    pts2_arr = np.float32(pts2conv)

    matrix = cv2.getPerspectiveTransform(pts1_arr, pts2_arr)
    result = cv2.warpPerspective(cutout_arr, matrix, (h, w))
    result = Image.fromarray(result.astype('uint8'), 'RGBA').rotate(180)
    result.save(d_dir + imidx + ".png")

def main():

    model_name='u2net'#u2netp
    img_name_list = glob.glob(image_dir + os.sep + '*')
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    if(model_name=='u2net'):
        net = U2NET(3,1)

    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main_path = "/home/tangoeye/Documents/Rebekah/slide_viewer/input_video/"
    video_path = main_path + "Freeland.mp4"
    image_dir = main_path + "Freeland"
    prediction_dir = main_path + "Freeland_fg/"
    model_dir = "/home/tangoeye/Documents/Rebekah/U-2-Net/saved_models/u2net/u2net.pth"
    vtoi = VideoToImages(video_path, image_dir + os.sep, 5, debug_mode=False)
    vtoi.convert()
    main()
    img_list = []
    shutil.rmtree(image_dir)
    for img in sorted(glob.glob(prediction_dir + "*.png")):
        img_name = int(img.split("/")[-1].split(".")[0])
        img_list.append(img_name)

    img_list.sort()

    images = []
    for i in img_list:
        image_file = prediction_dir + str(i) + ".png"
        images.append(imageio.imread(image_file))

    imageio.mimsave(prediction_dir + "Freeland.gif", images)




