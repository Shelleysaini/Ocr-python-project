# Import required libraries
import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Create EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Load the image
image = cv2.imread('picture_with_words.jpg')

# Check if the image was loaded correctly
if image is None:
    print("Error: Image not found or failed to load.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Optional preprocessing to improve OCR results
# _, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Perform OCR
results = reader.readtext(gray)

# Check if text was detected
if not results:
    print("No text found in the image.")
else:
    # Filter results by confidence (optional)
    CONFIDENCE_THRESHOLD = 0.3
    filtered_results = [r for r in results if r[2] >= CONFIDENCE_THRESHOLD]

    # Draw boxes and labels on the image
    for (bbox, text, prob) in filtered_results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = (int(top_left[0]), int(top_left[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Print results to terminal
    print("\nWords found in the picture (with confidence):")
    for (_, text, prob) in filtered_results:
        print(f"- {text} (I'm {prob*100:.2f}% sure)")

    print("\nJust the words:")
    for (_, text, _) in filtered_results:
        print(f"- {text}")

    # Save the extracted words to a .txt file
    with open("extracted_words.txt", "w") as file:
        file.write("Extracted Words:\n")
        for (_, text, _) in filtered_results:
            file.write(f"{text}\n")

    print("\nThe extracted words have been saved to 'extracted_words.txt'.")
