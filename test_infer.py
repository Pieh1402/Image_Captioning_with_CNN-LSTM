import os
import sys
from backend.inference import predict_from_path

if __name__ == "__main__":
    img_path = "archive/Images/1000268201_693b08cb0e.jpg"
    print("Beam width 1:", predict_from_path(img_path, beam_width=1))
    print("Beam width 3:", predict_from_path(img_path, beam_width=3))
