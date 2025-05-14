import RPi.GPIO as GPIO
import time
import cv2  # OpenCV for USB camera handling

# Setup GPIO pins for Shift Register
SER_PIN = 17
RCLK_PIN = 27
SRCLK_PIN = 22

# Setup GPIO for Buzzer
BUZZER_PIN = 18

# Setup GPIO for MQ135 sensor (digital output)
MQ135_PIN = 23

# Setup GPIO mode
GPIO.setmode(GPIO.BCM)

# Set GPIO pins as output
GPIO.setup(SER_PIN, GPIO.OUT)
GPIO.setup(RCLK_PIN, GPIO.OUT)
GPIO.setup(SRCLK_PIN, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(MQ135_PIN, GPIO.IN)

# Function to send data to the shift register (RGB LED control)
def shift_out(data):
    for i in range(8):
        GPIO.output(SER_PIN, (data >> i) & 1)
        GPIO.output(SRCLK_PIN, GPIO.HIGH)
        GPIO.output(SRCLK_PIN, GPIO.LOW)

# Function to handle the red LED (spoiled fruit)
def spoil_fruit_detected():
    # Turn on red LED
    shift_out(0b10000000)  # Set red LED pin high
    GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Activate buzzer
    time.sleep(1)
    GPIO.output(BUZZER_PIN, GPIO.LOW)  # Deactivate buzzer

# Function to handle the green LED (fresh fruit)
def fresh_fruit_detected():
    # Turn on green LED
    shift_out(0b01000000)  # Set green LED pin high

# Function to check the air quality (using MQ135 sensor)
def check_air_quality():
    # Simple digital read from MQ135
    if GPIO.input(MQ135_PIN) == GPIO.HIGH:
        return "spoiled"
    else:
        return "fresh"

# Function to capture an image from the USB camera using OpenCV
def capture_image():
    # Open the USB camera (0 is typically the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    
    # Capture one frame from the camera
    ret, frame = cap.read()

    if ret:
        # Save the captured image
        cv2.imwrite('captured_image.jpg', frame)
        print("Image captured and saved.")
    else:
        print("Error: Failed to capture image.")
    
    # Release the camera
    cap.release()

# Main function to check fruit freshness
def check_fruit_freshness():
    air_quality = check_air_quality()
    if air_quality == "spoiled":
        spoil_fruit_detected()
    else:
        fresh_fruit_detected()

    # Capture an image of the fruit
    capture_image()

# Loop through the process
while True:
    check_fruit_freshness()
    time.sleep(5)  # Check every 5 seconds

# Clean up GPIO after usage
GPIO.cleanup()
