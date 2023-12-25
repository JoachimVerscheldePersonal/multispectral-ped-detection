"""
Module that holds all the entities of the experiment
"""
from tqdm import tqdm
from typing import Any
import xml.etree.ElementTree as ET
import json
import re
import numpy as np
import cv2
import os
from PIL import Image

class MultispectralImage():
    """
    A multispectral image with 6 channels combining 3 channels from an optic image (red, green, blue) and 3 channels from a LWIR (Long-Wave InfraRed) image.
    """
   
    def __init__(self, optic_image_path: str, lwir_image_path: str) -> None:
        """
        Creates a multispectral image using a optic image and a lwir image

        Args:
            optic_image_path (str): the file path to the optic image 
            lwir_image_path (str): the file path to the lwir image
        """
        self.optic_image_path = optic_image_path
        self.lwir_image_path = lwir_image_path
        self.array = self.__build()


    def __build(self) -> np.ndarray:
        """
        Calculates the multispectral image by concatenating the thermal channels to the optic channels

        Returns:
            np.ndarray: returns a numpy array representing the multispectral image with the same size as the original image but with 6 channels
        """
        optic_image = cv2.imread(self.optic_image_path)
        lwir_image = cv2.imread(self.lwir_image_path)
        
        # Normalize LWIR image values to the range [0, 255]
        # lwir_image = cv2.normalize(lwir_image, None, 0, 255, cv2.NORM_MINMAX)
        multispectral_image = (optic_image + lwir_image) / 2
        return multispectral_image

    def write_to_file(self, file_path: str) -> None:
        """
        Writes the multispectral image to a file.

        Args:
            file_path (str): The file path where the multispectral image should be saved.
        """
        print(self.array.shape)
        print(self.array)
        image_file = Image.fromarray(np.uint8(self.array))
        image_file.save(file_path, format='TIFF')


class VOC2COCO:
    """
    This class converts the KAIST annotations that are in a custom format, closely related to the PASCAL VOC format, into the COCO annotation format.
    """
    def __init__(self) -> None:
        pass

    def convert(self,
                voc_annotations_paths: list[str],
                coco_annotations_output_file: str,
                night_prefix:bool = False) -> None:
        """
        Converts the input annotations to COCO format and saves them in the desired output folder

        Args:
            voc_annotations_paths (list[str]): a list of paths to the annotations
            coco_annotations_output_file (str): the path of the new coco annotation file
            night_prefix (bool, optional): Whether or not the night_ prefix has to be added to the output name. Defaults to False.
        """
        
        self.convert_xmls_to_cocojson(
            annotation_paths=voc_annotations_paths,
            label2id=self.get_label2id(),
            output_jsonpath=coco_annotations_output_file,
            night_prefix = night_prefix
        )

    def get_label2id(self) -> dict[str, int]:
        """
        Returns a list of unique identifiers, for every class there is 1 unique identifier

        Returns:
            dict[str, int]: dictionary with the classname as the key and the unique label identifier as the value
        """
        return {"person":1, "people":2, "cyclist":3}

    def get_image_info(self, annotation_root: ET.Element | Any, image_id: int, night_prefix: bool = False) -> dict[str , Any]:
        """
        Returns information about the image in COCO format

        Args:
            annotation_root (ET.Element | Any): the XML root of the annotation file
            image_id (int): the ID of the image
            night_prefix (bool, optional): Whether or not the night_ prefix has to be added to the output name. Defaults to False.

        Returns:
            dict[str , Any]: Returns a dictionary of information about the image in COCO format
        """
        voc_filename = annotation_root.findtext('filename')
        filename = voc_filename[voc_filename.rfind('/')+1:]
        size = annotation_root.find('size')
        width = int(size.findtext('width'))
        height = int(size.findtext('height'))

        image_info = {
            'file_name': filename if not night_prefix else f'night_{filename}',
            'height': height,
            'width': width,
            'id': image_id
        }
        return image_info


    def get_coco_annotation_from_obj(self, obj: ET.Element, label2id: dict[str,int]) -> dict[str, Any]:
        """
        Builds the full coco annotation based on the PASCAL VOC(ish) xml annotation

        Args:
            obj (ET.Element): the XML input annotation
            label2id (dict[str,int]): the dictionary containing the mapping between classnames and label id's

        Returns:
            dict[str, Any]: a dictionary containing the annotation in COCO format
        """
        label = obj.findtext('name')
        label = label.replace("?","")
        assert label in label2id, f"Error: {label} is not in label2id dictionary"
        category_id = label2id[label]
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.findtext('x')) - 1
        ymin = int(bndbox.findtext('y')) - 1
        xmax = xmin + int(bndbox.findtext('w'))
        ymax = ymin + int(bndbox.findtext('h'))
        assert xmax > xmin and ymax > ymin, f"Box size error: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
        o_width = xmax - xmin
        o_height = ymax - ymin
        ann = {
            'area': o_width * o_height,
            'iscrowd': 0,
            'bbox': [xmin, ymin, o_width, o_height],
            'category_id': category_id,
            'ignore': 0,
            'segmentation': [] 
        }
        return ann

    def convert_xmls_to_cocojson(self,
                                 annotation_paths: list[str],
                                 label2id: dict[str, int],
                                 output_jsonpath: str,
                                 night_prefix:bool = False):
        """
        Perfroms  the full conversion of the input annotations to the COCO format and saves them in the desired output location

        Args:
            annotation_paths (list[str]): the list of input annotations
            label2id (dict[str, int]): the dictionary containing the mapping between classnames and label id's
            output_jsonpath (str): the path of the destination json file that will contain the COCO annotations
            night_prefix (bool, optional): Whether or not the night_ prefix has to be added to the output name. Defaults to False.
        """
        output_json_dict = {
            "images": [],
            "type": "instances",
            "annotations": [],
            "categories": []
        }
        image_id = 1 
        annotation_id = 1
        print('Converting started')
        for annotation_path in tqdm(annotation_paths):
            # Read annotation xml
            annotation_tree = ET.parse(annotation_path)
            annotation_root = annotation_tree.getroot()

            img_info = self.get_image_info(annotation_root=annotation_root, image_id= image_id, night_prefix= night_prefix)
            output_json_dict['images'].append(img_info)

            for obj in annotation_root.findall('object'):
                ann = self.get_coco_annotation_from_obj(obj=obj, label2id=label2id)
                ann.update({'image_id': image_id, 'id': annotation_id})
                output_json_dict['annotations'].append(ann)
                annotation_id = annotation_id + 1

            image_id = image_id +1

        for label, label_id in label2id.items():
            category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
            output_json_dict['categories'].append(category_info)

        if not os.path.exists(os.path.dirname(output_jsonpath)):
            os.mkdir(os.path.dirname(output_jsonpath))

        if not os.path.exists(output_jsonpath):
            with open(output_jsonpath, 'w') as f:
                output_json = json.dumps(output_json_dict)
                f.write(output_json)
                print(f"Saved COCO annotations to {output_jsonpath}.")
        else:
            print(f"File {output_jsonpath} already exists.")

        