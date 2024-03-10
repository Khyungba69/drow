import cv2
import dlib
import imutils
import numpy as np
import smtplib
from imutils import face_utils
from kivy.app import App
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from scipy.spatial import distance

class DrowsinessApp(App):

    def build(self):
        self.box_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        # Header
        self.header = Label(text='Drowsiness Detection App', size_hint=(1, None), height=50, font_size=30,
                            color=(0, 0.6, 1, 1))
        self.box_layout.add_widget(self.header)

        # Camera and alarm selection
        self.create_camera_settings()

        # Log display
        self.create_log_display()

        # Status display
        self.create_status_display()

        # Image display
        self.create_image_display()

        # Control buttons
        self.create_control_buttons()

        # Initialize attributes
        self.flag = 0
        self.capture = None
        self.sound = None
        self.thresh = 0.25
        self.frame_check = 20
        self.drowsiness_threshold = 0.2  # Adjust as needed
        self.alert_frequency = 5  # Adjust as needed
        self.detect = dlib.get_frontal_face_detector()
        self.predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        self.lStart, self.lEnd = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        self.rStart, self.rEnd = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        self.mStart, self.mEnd = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        self.yawn_flag = False
        self.yawn_counter = 0
        self.yawn_threshold = 1  # Adjust as needed
        self.yawn_duration = 5  # Number of consecutive frames below threshold to detect a yawn

        return self.box_layout

    def create_camera_settings(self):
        settings_layout = BoxLayout(orientation='horizontal', spacing=30, size_hint_y=None, height=50)

        camera_label = Label(text='Camera Selection:', size_hint=(0.3, None), height=30, font_size=18,
                             color=(0.8, 0.8, 0.8, 1))
        self.camera_spinner = Spinner(text='Camera 0', values=('Camera 0', 'Camera 1'), size_hint=(0.7, None), height=30)
        settings_layout.add_widget(camera_label)
        settings_layout.add_widget(self.camera_spinner)

        alarm_label = Label(text='Alarm Sound Selection:', size_hint=(0.3, None), height=30, font_size=18,
                            color=(0.8, 0.8, 0.8, 1))
        self.alarm_spinner = Spinner(text='Alarm 1', values=('Alarm 1', 'Alarm 2'), size_hint=(0.7, None), height=30)
        settings_layout.add_widget(alarm_label)
        settings_layout.add_widget(self.alarm_spinner)

        self.box_layout.add_widget(settings_layout)

    def create_log_display(self):
        log_label = Label(text='Log:', size_hint=(1, None), height=30, font_size=24, color=(0.8, 0.8, 0.8, 1))
        self.log_text = TextInput(multiline=True, readonly=True, size_hint=(1, None), height=150)
        self.log_scroll = ScrollView(size_hint=(1, None), height=200)
        self.log_scroll.add_widget(self.log_text)

        self.box_layout.add_widget(log_label)
        self.box_layout.add_widget(self.log_scroll)

    def create_status_display(self):
        self.status_label = Label(text='Status: Waiting', size_hint=(1, None), height=30, font_size=24,
                                  color=(0.8, 0.8, 0.8, 1))
        self.ear_label = Label(text='EAR: 0.00', size_hint=(1, None), height=30, font_size=24,
                               color=(0.8, 0.8, 0.8, 1))
        self.mar_label = Label(text='MAR: 0.00', size_hint=(1, None), height=30, font_size=24,
                               color=(0.8, 0.8, 0.8, 1))
        self.box_layout.add_widget(self.status_label)
        self.box_layout.add_widget(self.ear_label)
        self.box_layout.add_widget(self.mar_label)

    def create_image_display(self):
        self.image = Image(size_hint=(1, 1))
        self.box_layout.add_widget(self.image)

    def create_control_buttons(self):
        button_layout = BoxLayout(orientation='horizontal', spacing=10, size_hint=(1, None), height=50)

        self.start_button = Button(text='Start', on_press=self.start_detection, background_color=(0, 0.7, 0.3, 1))
        self.stop_button = Button(text='Stop', on_press=self.stop_detection, background_color=(0.8, 0.2, 0.2, 1))
        self.settings_button = Button(text='Settings', on_press=self.open_settings,
                                      background_color=(0.3, 0.5, 0.8, 1))

        button_layout.add_widget(self.start_button)
        button_layout.add_widget(self.stop_button)
        button_layout.add_widget(self.settings_button)

        self.box_layout.add_widget(button_layout)

    def start_detection(self, instance):
        self.status_label.text = 'Status: Detecting'
        self.flag = 0
        self.sound = SoundLoader.load(f"{self.alarm_spinner.text}.wav")
        self.update_log("Detection started.")

        camera_index = int(self.camera_spinner.text[-1])
        self.capture = cv2.VideoCapture(camera_index)

        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def stop_detection(self, instance):
        self.status_label.text = 'Status: Stopped'
        Clock.unschedule(self.update)
        cv2.destroyAllWindows()
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        if self.sound is not None:
            self.sound.unload()
        self.sound = None
        self.update_log("Detection stopped.")

    def update_log(self, message):
        self.log_text.text += f'\n{message}'

    def open_settings(self, instance):
        settings_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        thresh_slider = Slider(min=0.1, max=0.4, value=self.thresh, step=0.01)
        frame_slider = Slider(min=5, max=40, value=self.frame_check, step=1)
        save_button = Button(text='Save', on_press=lambda x: self.save_settings(thresh_slider.value, frame_slider.value))

        settings_layout.add_widget(Label(text='Drowsiness Threshold', color=(0.8, 0.8, 0.8, 1)))
        settings_layout.add_widget(thresh_slider)
        settings_layout.add_widget(Label(text='Frame Check Count', color=(0.8, 0.8, 0.8, 1)))
        settings_layout.add_widget(frame_slider)
        settings_layout.add_widget(save_button)

        popup = Popup(title='Settings', content=settings_layout, size_hint=(None, None), size=(400, 300),
                      background_color=(0.2, 0.2, 0.2, 1))
        popup.open()

    def save_settings(self, thresh, frame_check):
        self.thresh = thresh
        self.frame_check = frame_check

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def mouth_aspect_ratio(self, mouth):
        A = distance.euclidean(mouth[1], mouth[7])
        B = distance.euclidean(mouth[2], mouth[6])
        C = distance.euclidean(mouth[3], mouth[5])
        mar = (A + B + C) / (3.0 * distance.euclidean(mouth[0], mouth[4]))
        return mar

    def update(self, dt):
        if not self.capture or not self.capture.isOpened():
            return

        ret, frame = self.capture.read()
        if ret:
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subjects = self.detect(gray)

            drowsiness_severity = 0
            consecutive_alerts = 0

            ear = 0  # Initialize ear here
            mar = 0  # Initialize mar here

            for subject in subjects:
                shape = self.predict(gray, subject)
                shape = face_utils.shape_to_np(shape)
                left_eye = shape[self.lStart:self.lEnd]
                right_eye = shape[self.rStart:self.rEnd]
                mouth = shape[self.mStart:self.mEnd]
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                mar = self.mouth_aspect_ratio(mouth)

                drowsiness_severity = max(drowsiness_severity, ear)

                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                mouth_hull = cv2.convexHull(mouth)  # Convex hull around mouth

                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)  # Draw contours around mouth

            if ear < self.thresh:
                self.flag += 1
                consecutive_alerts += 1
                if self.flag >= self.frame_check:
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "****************ALERT!****************", (10, 325),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if self.sound and not self.sound.state == 'play':
                        self.sound.play()
                        self.update_log("Drowsiness detected.")
                       

            else:
                self.flag = 0
                consecutive_alerts = 0
                if self.sound and self.sound.state == 'play':
                    self.sound.stop()

            if mar > 0.9:  # Adjust threshold as needed
                self.yawn_counter += 1
                if self.yawn_counter >= self.yawn_duration:
                    self.yawn_flag = True
            else:
                self.yawn_counter = 0
                self.yawn_flag = False

            self.ear_label.text = f'EAR: {drowsiness_severity:.2f}'
            self.mar_label.text = f'MAR: {mar:.2f}'

            if consecutive_alerts > 0:
                self.status_label.text = 'Status: Alert'
            elif self.yawn_flag:
                self.status_label.text = 'Status: Yawning'
                if self.sound and not self.sound.state == 'play':
                    self.sound.play()
                    self.update_log("Yawn detected.")

            frame = cv2.flip(frame, 0)
            buf = frame.tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def on_stop(self):
        if self.capture is not None:
            self.capture.release()
        if self.sound is not None:
            self.sound.unload()
        cv2.destroyAllWindows()

    

if __name__ == '__main__':
    DrowsinessApp().run()
