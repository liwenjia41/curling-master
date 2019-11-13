# -*- coding:utf-8 -*-
"""
Usage:
  # Image augment
  python preprocess.py --img_dir=/path/to/image/dir --direct=train\val
"""
import cv2
import numpy as np
import random
import xml.etree.ElementTree as ET
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, help='image directory')
# parser.add_argument('--serial', type=int, help='image name serial')
parser.add_argument('--direct', type=str, help='train or val')
args = parser.parse_args()


def read_xml(img_path):
    """Read bounding box from xml
    Args:
        img_path: path to image
    Return list of bounding boxes
    """
    anno_path = '.'.join(img_path.split('.')[:-1]) + '.xml'
    tree = ET.ElementTree(file=anno_path)
    root = tree.getroot()
    ObjectSet = root.findall('object')
    bboxes = []
    for object in ObjectSet:
        box = object.find('bndbox')
        x1 = int(box.find('xmin').text)
        y1 = int(box.find('ymin').text)
        x2 = int(box.find('xmax').text)
        y2 = int(box.find('ymax').text)
        bb = [x1, y1, x2, y2]
        bboxes.append(bb)
    return bboxes


def read_xml_no_truncated(img_path):
    """Read bounding box from xml with no truncated
    Args:
        img_path: path to image
    Return list of bounding boxes
    """
    anno_path = img_path.split('.')[0] + '.xml'
    tree = ET.ElementTree(file=anno_path)
    root = tree.getroot()
    ObjectSet = root.findall('object')
    bboxes = []
    for object in ObjectSet:
        truncated = int(object.find('truncated').text)
        if truncated==1:
            continue
        elif truncated==0:
            box = object.find('bndbox')
            x1 = int(box.find('xmin').text)
            y1 = int(box.find('ymin').text)
            x2 = int(box.find('xmax').text)
            y2 = int(box.find('ymax').text)
            bb = [x1, y1, x2, y2]
            bboxes.append(bb)
    return bboxes


def indent(elem, level=0):
    """Process xml indent.
    Args:
        elem: xml root node
        level: root node's level
    """
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def write_xml(save_path, boxes, size, direct='train'):
    """
    Generate xml file for new samples
    :param save_path: path to xml file
    :param boxes: bounding boxes
    :param size: image size
    :param direct: train/val
    """
    node_root = ET.Element('annotation')

    node_folder = ET.SubElement(node_root, 'folder')
    node_folder.text = direct

    node_filename = ET.SubElement(node_root, 'filename')
    node_filename.text = save_path.split('/')[-1] + '.jpg'

    node_path = ET.SubElement(node_root, 'path')
    node_path.text = save_path + '.jpg'

    node_source = ET.SubElement(node_root, 'source')
    node_dataset = ET.SubElement(node_source, 'dataset')
    node_dataset.text = 'Unknown'

    node_size = ET.SubElement(node_root, 'size')
    node_width = ET.SubElement(node_size, 'width')
    node_height = ET.SubElement(node_size, 'height')
    node_depth = ET.SubElement(node_size, 'depth')
    node_width.text = str(size)
    node_height.text = str(size)
    node_depth.text = '3'

    node_segmented = ET.SubElement(node_root, 'segmented')
    node_segmented.text = '0'

    for box in boxes:
        node_object = ET.SubElement(node_root, 'object')
        node_objName = ET.SubElement(node_object, 'name')
        node_objPose = ET.SubElement(node_object, 'pose')
        node_objTruncated = ET.SubElement(node_object, 'truncated')
        node_objDiff = ET.SubElement(node_object, 'difficult')
        node_objBnd = ET.SubElement(node_object, 'bndbox')
        node_xmin = ET.SubElement(node_objBnd, 'xmin')
        node_ymin = ET.SubElement(node_objBnd, 'ymin')
        node_xmax = ET.SubElement(node_objBnd, 'xmax')
        node_ymax = ET.SubElement(node_objBnd, 'ymax')
        node_objName.text = 'curling'
        node_objPose.text = 'Unspecified'
        node_objTruncated.text = '0'
        node_objDiff.text = '0'
        node_xmin.text = str(box[0])
        node_ymin.text = str(box[1])
        node_xmax.text = str(box[2])
        node_ymax.text = str(box[3])

    indent(node_root)
    tree = ET.ElementTree(node_root)
    save_path += '.xml'
    tree.write(save_path, encoding='utf-8')
    print(save_path, 'has been saved')


class ImageAugment:
    """
    Image Augment class.Also process Annotation xml.
    """
    def __init__(self, img_path):
        self.image = cv2.imread(img_path)
        self.boxes = read_xml(img_path)
        # self.boxes_no_truncated = read_xml_no_truncated(img_path)
        self.h, self.w, self.c = self.image.shape

    # create samples
    # def create_samples(self, bg_dir, serial=473):
    #     "bg_dir: background dir"
    #     for bg_file in os.listdir(bg_dir):
    #         bg_path = os.path.join(bg_dir, bg_file)
    #         bg_img = cv2.imread(bg_path)
    #         bg_img = cv2.resize(bg_img, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
    #         recreate_boxes = []
    #         n = len(self.boxes_no_truncated)
    #         if n==0:
    #             break
    #         print('length of boxes is: ', n)
    #         rand = random.randint(0,n-1)
    #         for i in range(rand+1):
    #             (x1,y1,x2,y2) = self.boxes_no_truncated[i]
    #             crop = self.image[y1:y2, x1:x2]
    #             crop_h, crop_w, _ = crop.shape
    #             left_x = random.randint(0, self.h-1)
    #             left_y = random.randint(0, self.w-1)
    #             #out of boundary
    #             while left_x+crop_h >= self.h or left_y+crop_w >= self.w:
    #                 left_x = random.randint(0, self.h - 1)
    #                 left_y = random.randint(0, self.w - 1)
    #             bg_img[left_x:left_x+crop_h, left_y:left_y+crop_w] = crop
    #             recreate_boxes.append([left_x, left_y, left_x+crop_h, left_y+crop_w])
    #         save_path = 'talor/curling-' + str(serial)
    #         cv2.imwrite(save_path+'.jpg', bg_img)
    #         write_xml(save_path, recreate_boxes)
    #         serial += 1

    # horizontal
    def horizontal(self):
        horizoned = cv2.flip(self.image, 1, None)
        hori_boxes = []
        for box in self.boxes:
            (x1, y1, x2, y2) = box
            x1_1 = self.w - x2
            x2_1 = self.w - x1
            hori_boxes.append([x1_1, y1, x2_1, y2])
        return horizoned, hori_boxes

    # vertically
    def vertically(self):
        verticalled = cv2.flip(self.image, 0, None)
        verti_boxes = []
        for box in self.boxes:
            (x1, y1, x2, y2) = box
            y1_1 = self.h - y2
            y2_1 = self.h - y1
            verti_boxes.append([x1, y1_1, x2, y2_1])
        return verticalled, verti_boxes

    # sharpen
    def sharpen(self):
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)
        sharpened = cv2.filter2D(self.image, -1, kernel)
        # bounding box not change
        return sharpened, self.boxes

    # add gaussian_noise
    def gauss_noise(self, mean=0, var=0.001):
        img = np.array(self.image/255, dtype=float)
        noise = np.random.normal(mean, var**0.5, img.shape)
        noise_added = img + noise
        if noise_added.min() < 0:
            low_clip = -1
        else:
            low_clip = 0
        noise_added = np.clip(noise_added, low_clip, 1.0)
        noise_added = np.uint8(noise_added*255)
        return noise_added, self.boxes

    # add salt_pepper_noise
    def salt_pepper_noise(self, prob=0.01):
        noise_added = np.zeros(self.image.shape, np.uint8)
        thres = 1-prob
        for i in range(self.h):
            for j in range(self.w):
                rdn = random.random()
                if rdn < prob:
                    noise_added[i][j] = 0
                elif rdn > thres:
                    noise_added[i][j] = 255
                else:
                    noise_added[i][j] = self.image[i][j]
        return noise_added, self.boxes

    # change brightness
    def brightness(self, gamma=40):
        blank = np.zeros([self.h, self.w, self.c], dtype=np.uint8)
        brighted = cv2.addWeighted(self.image, 1, blank, 2, gamma)
        return brighted, self.boxes

    # change contrast
    def contrast(self, alpha=1.3, gamma=0):
        blank = np.zeros([self.h, self.w, self.c], dtype=np.uint8)
        contrasted = cv2.addWeighted(self.image, alpha, blank, 2, gamma)
        return contrasted, self.boxes


if __name__ == '__main__':
    img_root_dir = args.img_dir
    # bg_root_dir = 'background'
    img_list = []
    file_list = os.listdir(img_root_dir)
    for file_path in file_list:
        if '.jpg' in file_path:
            img_list.append(file_path)
    # print('img_list:', img_list)

    # example for create sample
    # for img_file in img_list:
    #     print(serial)
    #     img_path = os.path.join(img_root_dir, img_file)
    #     imageAugment(img_path).create_samples(bg_root_dir, serial)
    #     serial += 31

    for img_file in img_list:
        # print(os.path.join(img_root_dir, img_file))
        img_aug = ImageAugment(os.path.join(img_root_dir, img_file))

        # ===================hirizontal=================
        horizoned, hori_boxes = img_aug.horizontal()
        hori_name = '.'.join(img_file.split('.')[:-1]) + '_(1)'
        hori_path = os.path.join(img_root_dir, hori_name)
        cv2.imwrite(hori_path + '.jpg', horizoned)
        write_xml(hori_path, hori_boxes, img_aug.h, args.direct)

        # ==================vertically=================
        vertically, verti_boxes = img_aug.vertically()
        verti_name = '.'.join(img_file.split('.')[:-1]) + '_(2)'
        verti_path = os.path.join(img_root_dir, verti_name)
        cv2.imwrite(verti_path+'.jpg', vertically)
        write_xml(verti_path, verti_boxes, img_aug.h, args.direct)

        # ===================sharpen====================
        sharpen, sharp_boxes = img_aug.sharpen()
        sharpen_name = '.'.join(img_file.split('.')[:-1]) + '_(3)'
        sharpen_path = os.path.join(img_root_dir, sharpen_name)
        cv2.imwrite(sharpen_path+'.jpg', sharpen)
        write_xml(sharpen_path, sharp_boxes, img_aug.h, args.direct)

        # ==================gaussian_noise==============
        gaussian_noise, gaussian_boxes = img_aug.gauss_noise(mean=0, var=0.01)
        gaussian_name = '.'.join(img_file.split('.')[:-1]) + '_(4)'
        gaussian_path = os.path.join(img_root_dir, gaussian_name)
        cv2.imwrite(gaussian_path+'.jpg', gaussian_noise)
        write_xml(gaussian_path, gaussian_boxes, img_aug.h, args.direct)

        # ================salt_pepper_noise=============
        sp_noise, sp_boxes = img_aug.salt_pepper_noise()
        sp_name = '.'.join(img_file.split('.')[:-1]) + '_(5)'
        sp_path = os.path.join(img_root_dir, sp_name)
        cv2.imwrite(sp_path+'.jpg', sp_noise)
        write_xml(sp_path, sp_boxes, img_aug.h, args.direct)

        # ==============brightness(1)crease=============
        brightness1, bright1_boxes = img_aug.brightness()
        bright1_name = '.'.join(img_file.split('.')[:-1]) + '_(6)'
        bright1_path = os.path.join(img_root_dir, bright1_name)
        cv2.imwrite(bright1_path+'.jpg', brightness1)
        write_xml(bright1_path, bright1_boxes, img_aug.h, args.direct)

        # =============brightness(2)decrease============
        brightness2, bright2_boxes = img_aug.brightness(gamma=-40)
        bright2_name = '.'.join(img_file.split('.')[:-1]) + '_(7)'
        bright2_path = os.path.join(img_root_dir, bright2_name)
        cv2.imwrite(bright2_path+'.jpg', brightness2)
        write_xml(bright2_path, bright2_boxes, img_aug.h, args.direct)

        # ===================contrast===================
        contrast, con_boxes = img_aug.contrast()
        con_name = '.'.join(img_file.split('.')[:-1]) + '_(8)'
        con_path = os.path.join(img_root_dir, con_name)
        cv2.imwrite(con_path+'.jpg', contrast)
        write_xml(con_path, con_boxes, img_aug.h, args.direct)

