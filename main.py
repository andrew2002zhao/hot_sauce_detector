import cv2

def take_image():
  cap = cv2.VideoCapture(0)

  while True:
    # open a camera feed
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Camera Feed', frame)
    
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
      # take a picture
     filename = f'images/captured_image.jpg'
     cv2.imwrite(filename, frame)
     print(f'Image saved as {filename}')
    

  cap.release()
  cv2.destroyAllWindows()
  return

  
# open a camera feed
take_image()

# train an AI model to classify between hot sauce and not hot sauce

# pass the picture to an AI model 

# ai model tells u whether its tapatio or not tapatio

