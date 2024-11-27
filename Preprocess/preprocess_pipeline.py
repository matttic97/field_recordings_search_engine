import os
import subprocess
import argparse


def run_source_separation(input_dir, output_dir):
    """
    Runs ZFTurbo's inference.py on all audio files in a directory.

    Args:
        input_dir (str): Path to the directory containing input audio files.
        output_dir (str): Directory to store the separated audio outputs.

    Returns:
        None
    """
    # Ensure the input and output directory exists
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Construct the command for ZFTurbo's inference.py
    command = [
        "python", "SourceSeparation/inference.py",  # Ensure inference.py is in the same directory or accessible
        "--input_folder", input_dir,
        "--store_dir", output_dir,
        "--extract_instrumental", False,
    ]

    try:
        # Run the inference.py command
        subprocess.run(command, check=True)
        print(f"Separation completed")
    except subprocess.CalledProcessError as e:
        print("An error occurred while processing")
    except FileNotFoundError:
        print("inference.py not found or not accessible. Ensure it's set up correctly.")

def run_asr(input_dir, output_dir, language="sl", model="turbo", file_extensions=("wav", "mp3", "flac")):
    """
    Runs the Whisper CLI on all audio files in a directory.

    Args:
        input_dir (str): Path to the directory containing input audio files.
        output_dir (str): Directory to store the transcription outputs.
        language (str): Language code for transcription (e.g., "en" for English, "sl" for Slovenian).
        model (str): Whisper model size (e.g., "tiny", "base", "small", "medium", "large").
        file_extensions (tuple): Tuple of allowed audio file extensions to process.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List all audio files in the input directory
    audio_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(file_extensions)
    ]

    if not audio_files:
        print(f"No audio files with extensions {file_extensions} found in {input_dir}.")
        return

    # Process each audio file
    for audio_file in audio_files:
        input_audio_path = os.path.join(input_dir, audio_file)
        print(f"Processing file: {input_audio_path}")

        # Construct the Whisper CLI command
        command = [
            "whisper", input_audio_path,
            "--output_dir", output_dir,
            "--language", language,
            "--model", model
        ]

        try:
            # Run the Whisper CLI command
            subprocess.run(command, check=True)
            print(f"Transcription completed for: {audio_file}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing {audio_file}: {e}")
        except FileNotFoundError:
            print("Whisper CLI is not installed or not found in PATH. Please install it first.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run source separation and ASR on audio files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing input audio files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory to store output files.")
    parser.add_argument("--text_output_dir", type=str, required=True, help="Path to the directory to store transcription files.")
    parser.add_argument("--language", type=str, default="sl", help="Language code for ASR (e.g., 'en' for English, 'sl' for Slovenian).")
    parser.add_argument("--model", type=str, default="turbo", help="Whisper model size (e.g., 'tiny', 'base', 'small', 'medium', 'large').")

    args = parser.parse_args()

    # Run source separation
    run_source_separation(args.input_dir, args.output_dir)

    # Run speech recognition
    run_asr(args.output_dir, args.text_output_dir, args.language, args.model)