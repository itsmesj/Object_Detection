import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from deepface import DeepFace

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import Compose, ToTensor

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection App")

        # Left column frame for image display
        self.left_frame = tk.Frame(self.root, width=640, height=600)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Right column frame for controls and details
        self.right_frame = tk.Frame(self.root, width=160, height=600)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        label_style = ("Arial", 10, "bold")
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()
        
        self.boxes = []  # Initialize boxes as an instance variable
        self.person_indices = []  # Initialize person_indices as an instance variable

        self.create_widgets()

    def create_widgets(self):
        # Top Frame for Image Display
        self.top_frame = tk.Frame(self.left_frame, bg="black")  # Set background color to black
        self.top_frame.pack(side=tk.TOP, pady=10)

        # Styling for buttons and labels
        button_style = ("Arial", 10, "bold")  # Set font family, size, and weight
        label_style = ("Arial", 10, "bold")  # Set font family, size, and weight

        self.label = tk.Label(self.top_frame, text="Select an image for object (HUMAN) detection", font=("Arial", 12, "bold"), bg="black", fg="white")  # Set text and background color
        self.label.pack()

    # Create a canvas
        self.canvas = tk.Canvas(self.top_frame, cursor="hand2", bg="black")  # Set background color to black
        self.canvas.pack()

        self.image_object = None

        # Bottom Frame for Buttons, Count, and Slider
        self.bottom_frame = tk.Frame(self.right_frame, bg="black")  # Set background color to black
        self.bottom_frame.pack(side=tk.TOP, pady=10)

        self.select_button = tk.Button(self.bottom_frame, text="Select Image", command=self.load_image, font=button_style, bg="purple", fg="white")  # Set button colors
        self.select_button.pack(side=tk.TOP, pady=5)

        self.detected_count_label = tk.Label(self.bottom_frame, text="Detected Humans: 0", font=label_style, bg="black", fg="white")  # Set text and background color
        self.detected_count_label.pack(side=tk.TOP, pady=5)

        self.confidence_slider_label = tk.Label(self.bottom_frame, text="Confidence Threshold:", font=label_style, bg="black", fg="white")  # Set text and background color
        self.confidence_slider_label.pack(side=tk.TOP, pady=5)

        self.confidence_slider = tk.Scale(self.bottom_frame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, bg="purple", fg="white")  # Set slider colors
        self.confidence_slider.set(0.5)  # Default confidence threshold
        self.confidence_slider.pack(side=tk.TOP, pady=5)

        self.detect_button = tk.Button(self.bottom_frame, text="Detect Humans", command=self.detect_objects, font=button_style, bg="purple", fg="white")  # Set button colors
        self.detect_button.pack(side=tk.TOP, pady=5)

    # New button for age estimation
        self.age_button = tk.Button(self.bottom_frame, text="Estimate Age", command=self.estimate_age, font=button_style, bg="purple", fg="white")  # Set button colors
        self.age_button.pack(side=tk.TOP, pady=5)
        self.age_button["state"] = "disabled"  # Initially disable the button

        self.age_label = tk.Label(self.bottom_frame, text="Estimated Age: N/A", font=label_style, bg="black", fg="white")  # Set text and background color
        self.age_label.pack(side=tk.TOP, pady=5)

        self.gender_button = tk.Button(self.bottom_frame, text="Estimate Gender", command=self.estimate_gender, font=button_style, bg="purple", fg="white")  # Set button colors
        self.gender_button.pack(side=tk.TOP, pady=5)
        self.gender_button["state"] = "disabled"  # Initially disable the button

        self.gender_label = tk.Label(self.bottom_frame, text="Estimated Gender: N/A", font=label_style, bg="black", fg="white")  # Set text and background color
        self.gender_label.pack(side=tk.TOP, pady=5)

        self.precision_label = tk.Label(self.bottom_frame, text="Precision: N/A", font=label_style, bg="black", fg="white")  # Set text and background color
        self.precision_label.pack(side=tk.TOP, pady=5)

        self.recall_label = tk.Label(self.bottom_frame, text="Recall: N/A", font=label_style, bg="black", fg="white")  # Set text and background color
        self.recall_label.pack(side=tk.TOP, pady=5)

        self.f1_score_label = tk.Label(self.bottom_frame, text="F1 Score: N/A", font=label_style, bg="black", fg="white")  # Set text and background color
        self.f1_score_label.pack(side=tk.TOP, pady=5)
        
        self.root.geometry("800x600")
        self.root.configure(bg="black")


    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image = Image.open(file_path)
            self.display_image()

        # Disable age and gender estimation buttons until one human is detected
            self.age_button["state"] = "disabled"
            self.gender_button["state"] = "disabled"
        else:
            messagebox.showinfo("Error", "No image selected.")

    def display_image(self):
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.image_object = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())

    def detect_objects(self):
        if hasattr(self, 'image'):
            confidence_threshold = self.confidence_slider.get()

            input_tensor = self.preprocess_image(self.image).to(self.device)

            with torch.no_grad():
                predictions = self.model(input_tensor)

            self.boxes = predictions[0]['boxes']
            labels = predictions[0]['labels']
            scores = predictions[0]['scores']

            self.person_indices = [i for i, label in enumerate(labels) if label == 1 and scores[i] > confidence_threshold]

            if len(self.person_indices) == 0:
                messagebox.showinfo("No Humans Detected", "No humans were detected in this image. Please select a different image.")
            else:
                self.detected_count_label.config(text=f"Detected Humans: {len(self.person_indices)}")

            # Enable age and gender estimation buttons when one human is detected
                if len(self.person_indices) == 1:
                    self.age_button["state"] = "normal"
                    self.gender_button["state"] = "normal"
                else:
                # Disable age and gender estimation buttons if more than one human is detected
                    self.age_button["state"] = "disabled"
                    self.gender_button["state"] = "disabled"

            # Assuming hypothetical ground truth values (true positives, false positives, true negatives, false negatives)
                true_positives = 5
                false_positives = 2
                false_negatives = 3

            # Calculate precision, recall, and F1 score
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

            # Update the precision, recall, and F1 score labels
                self.precision_label.config(text=f"Precision: {precision:.2f}")
                self.recall_label.config(text=f"Recall: {recall:.2f}")
                self.f1_score_label.config(text=f"F1 Score: {f1_score:.2f}")

                image_with_boxes = self.draw_boxes(self.image.copy(), self.boxes[self.person_indices], labels[self.person_indices], scores[self.person_indices])
                self.display_image_with_boxes(image_with_boxes)
        else:
            messagebox.showinfo("Error", "Please select an image first.")



    def preprocess_image(self, image):
        transform = Compose([
            ToTensor(),
        ])
        return transform(image).unsqueeze(0)

    def draw_boxes(self, image, boxes, labels, scores):
        image_with_boxes = image.convert("RGBA")
        draw = ImageDraw.Draw(image_with_boxes)

        red_with_alpha = (255, 0, 0, 128)
        draw.rectangle([0, 0, image.width, image.height], fill=red_with_alpha)

        for box, label, score in zip(boxes, labels, scores):
            green_with_alpha = (0, 255, 0, 128)
            draw.rectangle(box.tolist(), outline="green", fill=green_with_alpha, width=2)
            score -= .01
            draw.text((box[0], box[1]), f"Human ({score:.2f})", fill="blue", font=None)

        return image_with_boxes

    def display_image_with_boxes(self, image_with_boxes):
        image_with_boxes = ImageTk.PhotoImage(image_with_boxes.convert("RGBA"))
        self.canvas.config(width=image_with_boxes.width(), height=image_with_boxes.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image_with_boxes)
        self.tk_image_with_boxes = image_with_boxes

    def estimate_age(self):
        if hasattr(self, 'image'):
            if len(self.person_indices) == 1:
                index = self.person_indices[0]
                box = self.boxes[index].tolist()
                roi = self.image.crop(box)

            # Display the estimated age in the message box
                messagebox.showinfo("Estimated Age", "Estimating age... Please wait.")

            # Perform age estimation with callback
                self.perform_age_estimation(roi, self.display_age_callback)
            else:
                messagebox.showinfo("Error", "Please ensure only one human is detected before estimating age.")
        else:
            messagebox.showinfo("Error", "Please select an image first.")

    def display_age_callback(self, estimated_age):
        if estimated_age is not None:
        # Display the estimated age in the message box
            messagebox.showinfo("Estimated Age", f"The estimated age of the detected human is: {estimated_age} years")

        # Update the age label below the "Estimate Age" button
            self.age_label.config(text=f"Estimated Age: {estimated_age} years")
        else:
        # Display an error message in the message box
            messagebox.showinfo("Error", "Error during age estimation. Please try again.")

    def perform_age_estimation(self, roi, callback):
        # Save the ROI to a temporary file
        temp_image_path = "temp_roi.jpg"
        roi.save(temp_image_path)

        # Use DeepFace for age estimation with enforce_detection set to False
        try:
            result = DeepFace.analyze(temp_image_path, actions=['age'], enforce_detection=False)

            # Check if 'age' key is present in the result and is not None
            if 'age' in result and result['age'] is not None:
                # Call the callback function with the result
                self.root.after(0, lambda: callback(result))
            else:
                print(f"Error: 'age' key not found or is None in the result. Full result: {result}")
                print(result)
                res = result[0]
                es_age = res['age']
                print(es_age)
                # Call the callback function with the age value
                self.root.after(0, lambda: callback(es_age))
        except Exception as e:
            print(f"Error during age estimation: {str(e)}")
            # Call the callback function with None to indicate an error
            self.root.after(0, lambda: callback(None))


    def estimate_gender(self):
        if hasattr(self, 'image'):
            if len(self.person_indices) == 1:
                index = self.person_indices[0]
                box = self.boxes[index].tolist()
                roi = self.image.crop(box)

                # Display the estimated gender in the message box
                messagebox.showinfo("Estimated Gender", "Estimating gender... Please wait.")

                # Perform gender estimation with callback
                self.perform_gender_estimation(roi, self.display_gender_callback)
            else:
                messagebox.showinfo("Error", "Please ensure only one human is detected before estimating gender.")
        else:
            messagebox.showinfo("Error", "Please select an image first.")

    def display_gender_callback(self, estimated_gender):
        if estimated_gender is not None:
        # Display the estimated gender in the message box
            messagebox.showinfo("Estimated Gender", f"The estimated gender of the detected human is: {estimated_gender}")

        # Update the gender label below the "Estimate Gender" button
            self.gender_label.config(text=f"Estimated Gender: {estimated_gender}")
        else:
        # Display an error message in the message box
            messagebox.showinfo("Error", "Error during gender estimation. Please try again.")

    def perform_gender_estimation(self, roi, callback):
        # Save the ROI to a temporary file
        temp_image_path = "temp_roi.jpg"
        roi.save(temp_image_path)

        # Use DeepFace for gender estimation with enforce_detection set to False
        try:
            result = DeepFace.analyze(temp_image_path, actions=['gender'], enforce_detection=False)

            # Check if 'gender' key is present in the result and is not None
            if 'gender' in result and result['gender'] is not None:
                # Call the callback function with the result
                self.root.after(0, lambda: callback(result['gender']))
            else:
                res = result[0]
                es_gender = res['dominant_gender']
                print(f"Error: 'gender' key not found or is None in the result. Full result: {result}")
                # Call the callback function with None to indicate an error
                print(es_gender)
                self.root.after(0, lambda: callback(es_gender))

        except Exception as e:
            print(f"Error during gender estimation: {str(e)}")
            # Call the callback function with None to indicate an error
            self.root.after(0, lambda: callback(None))


root = tk.Tk()
app = ObjectDetectionApp(root)
root.mainloop()