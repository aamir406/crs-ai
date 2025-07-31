import cv2
import numpy as np
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
import openpyxl
import os

# Input params
print("Welcome to CRS AI Tracker! Please enter the following details.")
trans_no = input("Enter Company Trans No: ")
doc_no = input("Enter Company Doc No: ")
doc_title = input("Enter Document Title: ")

# Load the images
images = convert_from_path('images/test_plan.pdf')
os.makedirs('imgs/', exist_ok=True)

counter = 1
total_pages = len(images)
data = [] # to store all comments

# load CRS template
wb = openpyxl.load_workbook('resources/crs-template.xlsx')
template = wb.active

for page_no, page in enumerate(images, start=1):
    # for each page
    # print(f"page_no: {page_no}, page: {page}")
    print(f"Processing Page No: {page_no}")

    img_path = "imgs/page_{page_no}.png"
    page.save(img_path, "PNG")

    img = cv2.imread(img_path)

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define red color ranges in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)

    # Find contours
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Identify rectangular boxes
    boxes = []
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        if len(approx) == 4 and w * h > 1000:
            boxes.append((x, y, w, h))

    # Filter boxes with white background (low saturation, high value)
    valid_boxes = []
    for (x, y, w, h) in boxes:
        roi_hsv = hsv[y:y+h, x:x+w]
        avg = cv2.mean(roi_hsv)[:3]
        # print(f'page no: {page_no}, avg: {avg}')
        if avg[1] < 70 and avg[2] > 200:
            # print("Adding white box with red border")
            valid_boxes.append((x, y, w, h))

    # Prepare output image and results list
    output = img.copy()

    # OCR each valid box
    for i, (x, y, w, h) in enumerate(valid_boxes, 1):
        roi = img[y:y+h, x:x+w]
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Preprocess for OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # OCR
        text = pytesseract.image_to_string(thresh, config='--psm 6').strip()
        # print(f"Page No: {page_no}. Box Number: {i}. Extracted text: {text}")

        # Store results in CRS template

        # comments
        template[f'A{counter+11}'] = counter
        template[f'B{counter+11}'] = f"Page {page_no} of {total_pages}" # page no
        template[f'C{counter+11}'] = text # comment

        # metadata
        template['C3']= trans_no
        template['C4'] = doc_no
        template['C5'] = doc_title

        data.append({
            "No.": counter,
            "Page": f"Page {page_no} of {total_pages}",
            "Comment": text
        })

        counter += 1
    
    # Show image with rectangles
    cv2.imshow("Valid Boxes", output)
    cv2.waitKey(1500)
    cv2.destroyAllWindows()

print(pd.DataFrame(data))
wb.save("resources/extracted_comments.xlsx")

# # Export to Excel
# df = pd.DataFrame(data)
# df.to_excel("extracted_comments.xlsx", index=False)
# print("Exported to 'extracted_comments.xlsx'")


