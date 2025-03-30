# import cv2
# import numpy as np
# import pickle

# # Load the trained model
# with open("svm_ocr.pkl", "rb") as f:
#     model = pickle.load(f)

# # Load the input image
# image = cv2.imread("data/test2.jpg", cv2.IMREAD_GRAYSCALE)

# # Preprocess the image (resize to match training size)
# image = cv2.resize(image, (12, 12))  # Ensure it matches training size
# image = image.flatten().astype(np.float32) / 255.0  # Flatten & Normalize

# # Predict character
# predicted_char = model.predict([image])[0]

# # Ensure correct string conversion
# predicted_text = ""  # Initialize text variable
# predicted_text += str(predicted_char) + " "  # Convert int to str before concatenation

# print("Predicted Text:", predicted_text)

# import cv2
# import numpy as np

# # Load digit template
# template = cv2.imread("digit_template.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("digit_image.png", cv2.IMREAD_GRAYSCALE)

# # Apply template matching
# result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
# _, _, _, max_loc = cv2.minMaxLoc(result)

# # Draw rectangle around detected digit
# h, w = template.shape
# cv2.rectangle(img, max_loc, (max_loc[0] + w, max_loc[1] + h), 255, 2)

# cv2.imshow("Detected", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import pytesseract
from PIL import Image

img = Image.open("data/test2.jpg")
text = pytesseract.image_to_string(img, config="--oem 0 --psm 6")  # Legacy mode
print(text)


