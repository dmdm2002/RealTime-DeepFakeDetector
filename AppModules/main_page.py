import cv2
import queue
import numpy as np
import sounddevice as sd
from kivy.config import Config

from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import ScreenManager, Screen

from AppModules.Screens.GraphScreen import ScreenGraphManager
from AppModules.Screens.VideoScreen import ScreenVideoManager, ButtonDiv
from Detectors.DeepFake.inference import DeepFakeInference

# Config.set('graphics', 'width', '640')
# Config.set('graphics', 'height', '800')

# 오디오 데이터를 처리하기 위한 큐
audio_queue = queue.Queue()


class MainPage(GridLayout):
    # runs on initialization
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.deepfakedetector = DeepFakeInference()
        self.cols = 1
        # self.rows = 6
        self.spacing = (0, 0)  # rows와 cols 간의 간격을 0으로 설정
        self.padding = (10, 10, 10, 10)  # 레이아웃의 여백을 10으로 설정

        self.screen_video_manager = ScreenVideoManager(self.deepfakedetector)

        inside = GridLayout(cols=3, spacing=(5, 5), size_hint_y=None, height=80)
        inside.add_widget(ButtonDiv(self.screen_video_manager))
        self.add_widget(inside)
        self.add_widget(self.screen_video_manager)
        # ===============================================================================
        # 분할된 Screen manager part
        self.screen_graph_manager = ScreenGraphManager()
        self.add_widget(self.screen_graph_manager)
