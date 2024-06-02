from kivy.app import App
from kivy.config import Config
from kivy.uix.screenmanager import Screen, ScreenManager

from AppModules.main_page import MainPage

Config.set('graphics', 'width', '640')
Config.set('graphics', 'height', '800')


class EpicApp(App):
    def build(self):
        self.screen_manager = ScreenManager()
        self.connect_page = MainPage()
        screen = Screen(name="MainPage")
        screen.add_widget(self.connect_page)
        self.screen_manager.add_widget(screen)

        return self.screen_manager

if __name__ == "__main__":
    chat_app = EpicApp()
    chat_app.run()