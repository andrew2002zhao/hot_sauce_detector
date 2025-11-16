import cv2
from ultralytics import YOLO

image_location = "./images/captured_image.jpg"
model_weights_location = "pretrained/yolo11n.pt"
model_name = "yolo11n.pt"

def take_image():
  cap = cv2.VideoCapture(0)

  while True:
    # open a camera feed
    ret, frame = cap.read()
    # if ret:
    #     cv2.imshow('Camera Feed', frame)
    
    
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('q'):
    #     break
    # elif key == ord(' '):
    #   # take a picture
    filename = f'images/captured_image.jpg'
    cv2.imwrite(filename, frame)
    #  print(f'Image saved as {filename}')
    

  cap.release()
  cv2.destroyAllWindows()
  return

# open a camera feed
take_image()

# train an AI model to classify between hot sauce and not hot sauce
model = YOLO(model_name)
model.load(model_weights_location) 
# pass the picture to an AI model 
results = model(image_location)
# ai model tells u whether its tapatio or not tapatio
results[0].show()
