import os

from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.image import Image


class ScreenVoid(GridLayout):
    def __init__(self, screen_graph_manager):
        super().__init__()
        self.cols = 1
        self.rows = 2
        self.screen_graph_manager = screen_graph_manager
        fontName = '/'.join([os.getenv('SystemRoot'), '/fonts/gulim.ttc'])

        # 버튼 부분
        self.button = Button(text="자세히 버튼", size_hint_y=None, font_name=fontName, font_size=15)
        self.button.bind(on_press=self.change_to_detail)
        self.add_widget(self.button)

    def change_to_detail(self, instance):
        print('디테일 페이지로 넘어가기')
        self.screen_graph_manager.current = "screen_graph"


class ScreenGraph(GridLayout):
    def __init__(self, screen_graph_manager):
        super().__init__()
        self.cols = 1
        self.rows = 2
        self.screen_graph_manager = screen_graph_manager
        fontName = '/'.join([os.getenv('SystemRoot'), '/fonts/gulim.ttc'])

        # 버튼 부분
        self.button = Button(text="자세히 버튼", size_hint_y=None, font_name=fontName, font_size=15)
        self.button.bind(on_press=self.change_to_detail)
        self.add_widget(self.button)

        # 디테리 부분 (그래프 출력할 것)
        self.detail_info = Image(source="temp_image.jpg")
        self.add_widget(self.detail_info)

    def change_to_detail(self, instance):
        print('디테일 페이지로 넘어가기')
        self.screen_graph_manager.current = "screen_void"


def ScreenGraphManager():
    screen_graph_manager = ScreenManager()
    screen_void = Screen(name='screen_void')
    screen_void.add_widget(ScreenVoid(screen_graph_manager))
    screen_graph_manager.add_widget(screen_void)

    screen_graph = Screen(name='screen_graph')
    screen_graph.add_widget(ScreenGraph(screen_graph_manager))
    screen_graph_manager.add_widget(screen_graph)

    return screen_graph_manager
