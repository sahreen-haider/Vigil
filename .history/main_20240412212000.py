import os

from deepface import DeepFace


result = DeepFace.verify(img1_path = "PXL_20240208_185257753.NIGHT.jpg", img2_path = "PXL_20240208_185736526.NIGHT.jpg")

reso = DeepFace.verify(img1_path = "PXL_20240208_185257753.NIGHT.jpg", img2_path = "PXL_20240208_185736526.NIGHT.jpg")

print(result[threshold], reso[threshold])

