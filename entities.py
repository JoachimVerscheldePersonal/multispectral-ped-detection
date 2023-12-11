"""
Module that holds all the entities of the experiment
"""

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
        lwir_image = cv2.normalize(lwir_image, None, 0, 255, cv2.NORM_MINMAX)
        multispectral_image = cv2.merge(optic_image, lwir_image)
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