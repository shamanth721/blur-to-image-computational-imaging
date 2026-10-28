import cv2

# Open the default webcam.
camera = cv2.VideoCapture(0)

# Check whether the webcam opened successfully.
if not camera.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

print("Webcam started!")
print("Press S to save an image.")
print("Press Q to quit.")

while True:
    # Read one image from the webcam.
    success, frame = camera.read()

    if not success:
        print("Error: Could not capture an image.")
        break

    # Display the live webcam video.
    cv2.imshow("My First Computer Vision Project", frame)

    # Check whether a key was pressed.
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        # Save the current image to a file.
        cv2.imwrite("original.png", frame)
        print("Image saved as original.png")
        break

    elif key == ord("q"):
        print("Capture cancelled.")
        break

# Release the webcam and close the display window.
camera.release()
cv2.destroyAllWindows()