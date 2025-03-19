from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
results = model("test/004.jpg", conf=0.05)

# ดึงภาพที่ตรวจจับแล้ว
img_with_boxes = results[0].plot()

# บันทึกไฟล์เพื่อใช้บนเว็บ
cv2.imwrite("output.jpg", img_with_boxes)
