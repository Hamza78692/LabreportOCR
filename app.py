import cv2 as cv
import easyocr
import numpy as np
import streamlit as st
import pandas as pd

def main():
    st.title("OCR Extraction")

    # File uploader to accept an image
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = cv.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv.IMREAD_COLOR)

        # Convert the image to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        edge = cv.Canny(gray, 120, 230)

        # Apply thresholding
        _, thresh = cv.threshold(edge, 120, 255, cv.THRESH_BINARY)
        kernel = np.ones((9, 9), np.uint8)
        mor = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        st.image(mor , caption="Morphed", use_column_width=True)

        # Find contours
        contours, _ = cv.findContours(mor, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # Find the largest contour based on area
        largest_contour = max(contours, key=cv.contourArea)
        # Draw the contour on the original
        cv.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)

        st.image(image, caption="Original Image with Contour", use_column_width=True)
    
        # Create a mask of the largest contour
        mask = cv.drawContours(np.zeros_like(gray), [largest_contour], -1, 255, thickness=cv.FILLED)
    
        extracted_region = cv.bitwise_and(image, image, mask=mask)

        # Use EasyOCR to find text within the largest contour
        reader = easyocr.Reader(['en'])
        result = reader.readtext(extracted_region, detail=0)

        # Organize results into rows
        rows = [result[i:i+7] for i in range(0, len(result), 7)]

        # Display the extracted region
        st.image(extracted_region, caption="Extracted Region", use_column_width=True)

        # Create a pandas DataFrame from the extracted text
        df = pd.DataFrame(rows[1:], columns=result[0:7])

        # Display the DataFrame
        st.dataframe(df)

        # Save the DataFrame as a CSV file
        df.to_csv('output.csv', index=False)

if __name__ == "__main__":
    main()
