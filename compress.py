import subprocess
import os
import json
import argparse

def compress_video(input_path, output_path):
    """
    Compresses a video to approximately half its original size.

    This function uses ffmpeg to re-encode a video with a target bitrate
    calculated to reduce the file size by about 50%.

    Args:
        input_path (str): The path to the input video file.
        output_path (str): The path where the compressed video will be saved.

    Returns:
        bool: True if compression was successful, False otherwise.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return False

    print(f"Starting compression for: {input_path}")

    try:
        # 1. Get video duration and other format info using ffprobe
        # ffprobe is a tool that comes with ffmpeg for analyzing media streams.
        ffprobe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            input_path
        ]
        
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
        media_info = json.loads(result.stdout)
        
        duration = float(media_info['format']['duration'])
        
        # 2. Get the original file size in bytes
        original_size_bytes = os.path.getsize(input_path)
        
        # 3. Calculate the target total bitrate for a 50% smaller file
        # Target size = original_size / 2
        # Target bitrate (bits/sec) = (Target size in bytes * 8) / duration
        target_total_bitrate = (original_size_bytes * 8 / 2) / duration
        
        # 4. Set a reasonable audio bitrate (e.g., 128 kbps) and calculate video bitrate
        # We subtract the audio bitrate from the total to find the video bitrate.
        audio_bitrate = 128000  # 128 kbps in bits per second
        target_video_bitrate = target_total_bitrate - audio_bitrate
        
        if target_video_bitrate <= 0:
            print("Error: Target video bitrate is too low or negative.")
            print("This can happen with short videos or files with high-quality audio.")
            print("Consider using a lower audio bitrate or different compression method.")
            return False

        print(f"Original file size: {original_size_bytes / 1024 / 1024:.2f} MB")
        print(f"Video duration: {duration:.2f} seconds")
        print(f"Target total bitrate: {target_total_bitrate / 1000:.0f} kbps")
        print(f"Target video bitrate: {target_video_bitrate / 1000:.0f} kbps")
        print(f"Audio bitrate: {audio_bitrate / 1000:.0f} kbps")

        # 5. Construct and run the ffmpeg command for compression
        # -i: input file
        # -b:v: target video bitrate
        # -b:a: target audio bitrate
        # -y: overwrite output file if it exists
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', input_path,
            '-b:v', f'{int(target_video_bitrate)}',
            '-b:a', f'{audio_bitrate}',
            '-y',
            output_path
        ]
        
        print("\nRunning ffmpeg... This may take a while.")
        print(f"Command: {' '.join(ffmpeg_cmd)}")
        
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        
        # 6. Verify the result
        compressed_size_bytes = os.path.getsize(output_path)
        reduction_percent = (1 - compressed_size_bytes / original_size_bytes) * 100
        
        print("\nCompression complete!")
        print(f"Output file saved to: {output_path}")
        print(f"New file size: {compressed_size_bytes / 1024 / 1024:.2f} MB")
        print(f"File size reduced by: {reduction_percent:.2f}%")
        
        return True

    except FileNotFoundError:
        print("\nError: 'ffmpeg' or 'ffprobe' not found.")
        print("Please make sure ffmpeg is installed and accessible in your system's PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print("\nAn error occurred during ffmpeg/ffprobe execution.")
        print(f"Return code: {e.returncode}")
        print(f"Error output:\n{e.stderr.decode('utf-8')}")
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return False

if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Compress a video to approximately half its original file size.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input", help="Path to the input video file.")
    parser.add_argument("output", help="Path for the compressed output video file.")
    
    args = parser.parse_args()
    
    # Run the compression function with the provided arguments
    compress_video(args.input, args.output)
