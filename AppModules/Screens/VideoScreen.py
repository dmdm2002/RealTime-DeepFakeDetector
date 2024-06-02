import os
import cv2
import numpy as np
import sounddevice as sd
import PIL.Image as PILImage

from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.image import Image
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.graphics.texture import Texture


class ButtonDiv(GridLayout):
    def __init__(self, screen_video_manager):
        super().__init__()
        self.cols = 3
        self.rows = 1
        self.screen_video_manager = screen_video_manager
        fontName = '/'.join([os.getenv('SystemRoot'), '/fonts/gulim.ttc'])

        self.button_cam = Button(text="캠 버튼", size_hint_y=None, font_name=fontName, font_size=15)
        self.button_cam.bind(on_press=self.press_button_cam)

        self.button_deepfake = Button(text="딥페이크 영상 버튼", size_hint_y=None, font_name=fontName, font_size=15)
        self.button_deepfake.bind(on_press=self.press_button_deepfake)

        self.button_deepvoice = Button(text="딥보이스 영상 버튼", size_hint_y=None, font_name=fontName, font_size=15)
        self.button_deepvoice.bind(on_press=self.press_button_deepvoice)

        self.add_widget(self.button_cam)
        self.add_widget(self.button_deepfake)
        self.add_widget(self.button_deepvoice)

    def press_button_cam(self, instance):
        self.screen_video_manager.current = 'screen_cam'

    def press_button_deepfake(self, instance):
        self.screen_video_manager.current = 'screen_video'

    def press_button_deepvoice(self, instance):
        self.screen_video_manager.current = 'screen_voice'


class ScreenCAM(GridLayout):
    def __init__(self, deepfakedetector):
        super().__init__()
        self.cols = 1
        self.rows = 3
        self.deepfakedetector = deepfakedetector

        fontName = '/'.join([os.getenv('SystemRoot'), '/fonts/gulim.ttc'])

        # self.add_widget(ButtonDiv(self.screen_video_manager))

        # 딥페이크 탐지 출력값
        self.prob_label = Label(text="딥페이크 확률: 0%", size_hint_y=None, font_name=fontName)
        self.add_widget(self.prob_label)

        # 캠 출력
        try:
            self.cam = Camera(play=True, size_hint_y=None, resolution=(224, 224))
            self.add_widget(self.cam)
        except:
            self.detail_info = Image(source="temp_image.jpg")
            self.add_widget(self.detail_info)

        # 음성 입력
        # stream = sd.Stream(callback=self.audio_callback)
        # stream.start()

        # Detection 진행
        # Clock.schedule_interval(self.update, 1.0 / 2.0)

    def audio_callback(self, indata, outdata, frames, time, status):
        volume_norm = np.linalg.norm(indata)
        # indata를 outdata에 넣으면 마이크로 넘어온 데이터가 스피커로 출력된다.
        outdata[:] = indata
        # self.audio_label.text = f"Audio Level: {volume_norm:.2f}"

    # def get_cam_frame_predict(self, dt):
    #     texture = self.cam.texture
    #     size = texture.size
    #     pixels = texture.pixels
    #     pil_image = PILImage.frombytes(mode='RGB', size=size, data=pixels)
    #     output = self.deepfakedetector.run(pil_image)
    #
    #     self.prob_label.text = f"딥페이크 확률: {(1 - output) * 100:.2f}%"
    #
    #     f = open(f"cam_detection_score.txt", 'w', encoding='utf-8')
    #     f.write(
    #         f'{output}\n')
    #     f.close()


class ScreenVideo(GridLayout):
    def __init__(self, deepfakedetector):
        super().__init__()
        self.cols = 1
        self.rows = 2
        self.deepfakedetector = deepfakedetector
        fontName = '/'.join([os.getenv('SystemRoot'), '/fonts/gulim.ttc'])

        # 딥페이크 탐지 출력값
        self.output_layout = GridLayout(cols=2, spacing=(5, 5), size_hint_y=None, height=80)
        self.prob_label = Label(text="딥페이크 확률: 0%", size_hint_y=None, font_name=fontName)
        self.detection_result = Label(text="위조 위험", size_hint_y=None, font_name=fontName)

        self.output_layout.add_widget(self.prob_label)
        self.output_layout.add_widget(self.detection_result)

        # OpenCV 비디오 캡처 객체
        self.cap = cv2.VideoCapture('D:/Side/CodeGate/Dataset/sample_video/fakevideo.mp4')

        # Kivy Image 위젯
        self.img = Image()

        self.add_widget(self.output_layout)
        self.add_widget(self.img)

        # Detection 진행
        Clock.schedule_interval(self.get_frame_predict, 1.0 / 2.0)
        # CV로 불러온 영상 출력 Frame
        Clock.schedule_interval(self.get_frame_video, 1.0 / 30.0)

    def get_frame_predict(self, dt):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(frame_rgb)
            output = self.deepfakedetector.run(pil_image)
            self.prob_label.text = f"딥페이크 확률: {(1 - output) * 100:.2f}%"

            if ((1 - output) * 100) < 75:
                self.detection_result.text = "위조 안전"
            else:
                self.detection_result.text = "위조 위험"

            with open("deep_fake_video_score.txt", 'a', encoding='utf-8') as f:
                f.write(f'{output}\n')

    def get_frame_video(self, dt):
        ret, frame = self.cap.read()
        if ret:
            # Convert to Kivy texture
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img.texture = texture
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


def ScreenVideoManager(deepfakedetector):
    screen_video_manager = ScreenManager()
    screen_can = Screen(name='screen_cam')
    screen_can.add_widget(ScreenCAM(deepfakedetector))
    screen_video_manager.add_widget(screen_can)

    screen_video = Screen(name='screen_video')
    screen_video.add_widget(ScreenVideo(deepfakedetector))
    screen_video_manager.add_widget(screen_video)

    # screen_voice = Screen(name='screen_voice')
    # screen_video.add_widget(Screen())
    # screen_video_manager.add_widget(screen_video)

    return screen_video_manager

