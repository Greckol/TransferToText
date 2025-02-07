import os
import sys
import whisper
import time


def seconds_to_srt_time(seconds):
    """
    Преобразует секунды в формат времени для SRT: HH:MM:SS,mmm.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def generate_srt(segments, output_path):
    """
    Создает SRT-файл на основе сегментов транскрипции.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start = seconds_to_srt_time(segment["start"])
            end = seconds_to_srt_time(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")


def process_audio_file(audio_file, model):
    print(f"\nОбработка файла: {audio_file}")

    if not os.path.exists(audio_file):
        print(f"Файл не найден: {audio_file}")
        return

    # Создаем папку для результатов (имя файла без расширения)
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    folder_path = os.path.join(os.path.dirname(audio_file), base_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Создана директория: {folder_path}")

    print("Запуск транскрипции...")
    start_time = time.time()

    # Выполнение транскрипции (язык - русский)
    result = model.transcribe(audio_file, language="ru")

    end_time = time.time()
    print(f"Транскрипция завершена за {end_time - start_time:.2f} секунд.")

    # Сохранение результата в формате TXT
    txt_output = os.path.join(folder_path, f"{base_name}.txt")
    with open(txt_output, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"TXT сохранён: {txt_output}")

    # Сохранение результата в формате SRT
    srt_output = os.path.join(folder_path, f"{base_name}.srt")
    generate_srt(result["segments"], srt_output)
    print(f"SRT сохранён: {srt_output}")


def main():
    if len(sys.argv) < 2:
        print("Использование: python transcribe_video.py audio1.mp3 audio2.mp3 ...")
        sys.exit(1)

    print("Загрузка модели Whisper (medium) на устройстве 'cuda' ...")
    model = whisper.load_model("medium", device="cuda")
    print("Модель загружена.")

    for audio_file in sys.argv[1:]:
        process_audio_file(audio_file, model)


if __name__ == "__main__":
    main()
