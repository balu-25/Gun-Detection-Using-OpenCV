import cv2
import imutils
import datetime
import time
import yagmail
import os

# --- Email info (already provided) ---
SENDER_EMAIL = "pramidibalu25@gmail.com"
RECEIVER_EMAIL = "pramidibalu2005@gmail.com"
SUBJECT = "⚠️ Gun Detection Alert!"

# --- Initialize yagmail (uses credentials stored via yagmail.register) ---
# Option A (recommended if you already ran yagmail.register):
yag = yagmail.SMTP(SENDER_EMAIL)

# Option B (uncomment to pass password directly -- less secure):
# yag = yagmail.SMTP(SENDER_EMAIL, "YOUR_GMAIL_PASSWORD")

def send_email_alert(frame):
    """
    Save a snapshot and send an email with the image attached.
    """
    try:
        # Save snapshot with timestamp
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gun_detected_{ts}.jpg"
        cv2.imwrite(filename, frame)

        body = f"""
        🚨 Gun detected by security camera!

        Time: {datetime.datetime.now().strftime("%d %b %Y %I:%M:%S %p")}
        Please check the attached image.
        """

        # send email with attachment
        yag.send(
            to=RECEIVER_EMAIL,
            subject=SUBJECT,
            contents=body,
            attachments=filename
        )

        print(f"📧 Email alert sent and image saved as {filename}.")

        # optional: remove the saved image after sending to save disk space
        try:
            os.remove(filename)
        except Exception:
            pass

    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# --- Gun detection setup ---
gun_cascade = cv2.CascadeClassifier('cascade.xml')
camera = cv2.VideoCapture(0)

if gun_cascade.empty():
    print("Error: Cascade file not found or failed to load.")
    exit()

gun_detected_frames = 0
detection_threshold = 5  # require detection in 5 continuous frames
gun_exist = False

print("Starting camera... Press 'q' to quit.")

while True:
    ret, frame = camera.read()
    if not ret or frame is None:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    guns = gun_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(120, 120)
    )

    if len(guns) > 0:
        gun_detected_frames += 1
    else:
        gun_detected_frames = 0

    if gun_detected_frames >= detection_threshold:
        gun_exist = True

    for (x, y, w, h) in guns:
        if w * h > 25000:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Gun Detected", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(frame, datetime.datetime.now().strftime("%d %b %Y %I:%M:%S %p"),
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    cv2.imshow("Security Feed", frame)

    if gun_exist:
        print("⚠️ Gun confirmed detected in multiple frames!")
        # send email with the current frame attached
        send_email_alert(frame)
        time.sleep(1)
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
