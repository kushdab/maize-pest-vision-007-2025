import cv2
import numpy as np
import argparse
import os

class MaizePestDetector:
    def __init__(self, sensitivity=0.5):
        # Sensitivity 0.0 to 1.0
        self.sensitivity = sensitivity

    def process_image(self, image_path):
        if not os.path.exists(image_path):
            return None, "File not found"

        image = cv2.imread(image_path)
        if image is None:
            return None, "Invalid image format"

        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for green (healthy leaf)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        # Create a mask to isolate the leaf
        mask = cv2.inRange(hsv, lower_green, upper_green)
        leaf_only = cv2.bitwise_and(image, image, mask=mask)

        # Convert to grayscale to find holes/damage inside the leaf area
        gray = cv2.cvtColor(leaf_only, cv2.COLOR_BGR2GRAY)
        
        # Inverse thresholding to find non-green/damaged spots within the leaf
        # FAW damage often appears as light brown spots or transparent windows
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours of damage
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        output_img = image.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter by area size based on sensitivity
            if 100 < area < (50000 * self.sensitivity):
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append((x, y, w, h))
                
                # Draw rectangle and label
                cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(output_img, "FAW Damage", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return output_img, detections

def main():
    parser = argparse.ArgumentParser(description="Identify Fall Armyworm damage on maize leaves.")
    parser.add_argument("--input", required=True, help="Path to the input image")
    parser.add_argument("--output", default="result.jpg", help="Path to save the result")
    parser.add_argument("--sensitivity", type=float, default=0.5, help="Detection sensitivity (0.1 - 1.0)")
    
    args = parser.parse_args()
    
    detector = MaizePestDetector(sensitivity=args.sensitivity)
    print(f"Processing: {args.input}...")
    
    result_img, data = detector.process_image(args.input)

    if result_img is not None:
        if isinstance(data, list):
            count = len(data)
            print(f"Analysis Complete: Found {count} potential damage spots.")
            cv2.imwrite(args.output, result_img)
            print(f"Result saved to: {args.output}")
            
            # Show result if GUI is available
            try:
                cv2.imshow("Detection Result", result_img)
                print("Press any key to close window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except:
                pass
        else:
            print(f"Error: {data}")
    else:
        print("Error: Could not process image.")

if __name__ == "__main__":
    # Example usage: python detect.py --input leaf.jpg
    main()
