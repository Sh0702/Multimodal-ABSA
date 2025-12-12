import argparse
import os
import glob
import whisper
import json
from pathlib import Path

# Uses ffmpeg for video/audio handling

def transcribe_file(model, input_path, output_folder):
    result = model.transcribe(str(input_path), verbose=False)
    basename = Path(input_path).stem
    json_path = Path(output_folder) / f"{basename}.json"
    srt_path = Path(output_folder) / f"{basename}.srt"
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    # Optionally save srt
    with open(srt_path, 'w') as f:
        for segment in result["segments"]:
            f.write(f"{segment['id']+1}\n{segment['start']:.2f} --> {segment['end']:.2f}\n{segment['text']}\n\n")
    print(f"Saved: {json_path} and {srt_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch speech2text using Whisper (medium) on audio/video files.")
    parser.add_argument('--input_folder', type=str, required=True, help='Path to folder of audio/video files')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to output subtitles (JSON/SRT)')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    model = whisper.load_model('medium')

    for ext in ('*.wav', '*.mp3', '*.m4a', '*.mp4', '*.mov', '*.avi'):
        for audio_path in glob.glob(os.path.join(args.input_folder, ext)):
            transcribe_file(model, audio_path, args.output_folder)

if __name__ == "__main__":
    main()
