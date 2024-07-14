from tkinter import *
from PIL import Image, ImageTk
import cv2
import tkinter as tk
import threading
import time
from ultralytics import YOLO
import numpy as np
import os
if '__file__' in globals():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
else:
    current_dir = os.getcwd()
r = Tk()
r.geometry('1600x750')
r.title("Helmet Detector")

cap = cv2.VideoCapture('vid.mp4')

model = YOLO('best.pt')

with open("coco1.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

captured_helmets = []
count = 0
running = False
fps = 0
frame_count = 0
prev_time = time.time()
previous_boxes = None
last_detection_time = time.time()
def start():
    global running
    if not running:
        running = True
        threading.Thread(target=update_frame, daemon=True).start()

def end():
    global running
    running = False

def is_similar_box(new_box, existing_boxes, threshold=200):
    for box in existing_boxes:
        if all(abs(new_box[i] - box[i]) < threshold for i in range(4)):
            return True
    return False

def display_images(box):
    img_no_helmet_path = "img_no_helmet"
    row_count = 0
    col_count = 0
    for filename in os.listdir(img_no_helmet_path):
        file_path = os.path.join(img_no_helmet_path, filename)
        img = Image.open(file_path)
        img = img.resize((int(img.width * 0.65), int(img.height * 0.65)))
        img_tk = ImageTk.PhotoImage(img)

        canvas = Canvas(box, width=img.width, height=img.height + 20)  
        canvas.grid(row=row_count, column=col_count, padx=28, pady=10)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.image = img_tk
        canvas.create_text(img.width / 2, img.height + 10, text=filename, fill="black", font=("Arial", 7), anchor="center")

        col_count += 1
        if col_count == 6:  
            col_count = 0
            row_count += 1

def calculate_fps():
    global frame_count, prev_time
    current_time = time.time()
    elapsed_time = current_time - prev_time
    print(frame_count)
    print(elapsed_time)
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
    else:
        fps = 0
    prev_time = current_time
    frame_count = 0
    return fps

def are_boxes_similar(boxes1, boxes2, threshold=200):
    if boxes1 is None or boxes2 is None:
        return False
    if len(boxes1) != len(boxes2):
        return False
    for box1, box2 in zip(boxes1, boxes2):
        if not all(abs(box1[i] - box2[i]) < threshold for i in range(4)):
            return False
    return True

def update_frame():
    global captured_helmets, count, fps, frame_count, prev_time, previous_boxes, canvas

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (box1.winfo_width(), box1.winfo_height()))
        
        if frame_count % 20 == 0:
            results = model.predict(frame)
            a = results[0].boxes.data
            a = np.array(a)

            if are_boxes_similar(a, previous_boxes):
                frame_count += 3
                continue
            
            previous_boxes = a

            directory = 'img_no_helmet'
            for box in a:
                x1, y1, x2, y2, _, d = box.astype(int)
                c = class_list[d]
                x1_scaled = int(x1 * frame_resized.shape[1] / frame_rgb.shape[1])
                y1_scaled = int(y1 * frame_resized.shape[0] / frame_rgb.shape[0])
                x2_scaled = int(x2 * frame_resized.shape[1] / frame_rgb.shape[1])
                y2_scaled = int(y2 * frame_resized.shape[0] / frame_rgb.shape[0])

                if d == 0:
                    cv2.rectangle(frame_resized, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (255, 0, 0), 1)
                    cv2.putText(frame_resized, f'{c}', (x1_scaled, y1_scaled - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                elif d == 1:
                    cv2.rectangle(frame_resized, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 0, 255), 1)
                    cv2.putText(frame_resized, f'{c}', (x1_scaled, y1_scaled - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    if not is_similar_box((x1, y1, x2, y2), captured_helmets):
                        helmet_img = frame_rgb[y1:y2, x1:x2]
                        resize = cv2.resize(helmet_img, (128, 128))
                        resize_rgb = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
                        file_path = f"{directory}/pic_{count}.jpg"
                        cv2.imwrite(file_path, resize_rgb)
                        captured_helmets.append((x1, y1, x2, y2))
                        count += 1
                        display_images(box3)

        image = Image.fromarray(frame_resized)
        bg = ImageTk.PhotoImage(image)
        fps_label.config(text=f"FPS: {fps:.2f}")
        canvas.create_image(0, 0, anchor=tk.NW, image=bg)
        canvas.image = bg
        frame_count += 1
        fps = calculate_fps()
    

box1 = LabelFrame(r, text="Video Capture")
box2 = LabelFrame(r, text="Option")
box3 = LabelFrame(r, text="Image")

r.grid_rowconfigure(0, weight=0)   
r.grid_rowconfigure(1, weight=1)  
r.grid_columnconfigure(0, weight=0)

image = Image.open("helmet.png")
image = image.resize((int(image.width * 1.25), int(image.height * 1.25)))
bg = ImageTk.PhotoImage(image)

canvas = Canvas(box1, width=image.width, height=image.height)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, anchor=NW, image=bg)

box1.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
box2.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
box3.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=10, pady=10)

bttn_start = Button(box2, text="Start", font="arial 10", width=10, command=start)
bttn_start.grid(row=0, column=0, padx=10, pady=10)

bttn_stop = Button(box2, text="Stop", font="arial 10", width=10, command=end)
bttn_stop.grid(row=1, column=0, padx=10, pady=10)

fps_label = Label(box2, text="FPS: 0.00", font="arial 10")
fps_label.grid(row=2, column=0, padx=10, pady=10)
r.mainloop()
