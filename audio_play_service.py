import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pydub import AudioSegment
from pydub.playback import play
import time

class AudioHandler(FileSystemEventHandler):
    def __init__(self, audio_folder):
        self.audio_folder = audio_folder
        self.audio_files = []
        self.is_playing = False
        print(f"Monitoring folder: {self.audio_folder}")

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.mp3', '.wav', '.ogg')):
            print(f"New audio file detected: {event.src_path}")
            self.audio_files.append(event.src_path)
            if not self.is_playing:
                self.play_audio()

    def play_audio(self):
        while self.audio_files:
            self.is_playing = True
            audio_file = self.audio_files.pop(0)  # 取出队列中的第一个文件
            print(f"Playing audio file: {audio_file}")
            audio = AudioSegment.from_file(audio_file)
            play(audio)
        self.is_playing = False

def main():
    audio_folder = "./output"
    event_handler = AudioHandler(audio_folder)
    observer = Observer()
    observer.schedule(event_handler, audio_folder, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()

