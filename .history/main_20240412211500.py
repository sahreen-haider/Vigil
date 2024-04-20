import os

from deepface import DeepFace


result = DeepFace.verify(img1_path = "PXL_20240208_185257753.NIGHT.jpg", img2_path = "PXL_20231104_072540596.PORTRAIT.jpg")

print(result)

