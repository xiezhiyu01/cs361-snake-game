import os
import imageio.v2 as imageio
from pathlib import Path
import re
import argparse

def main():
    parser = argparse.ArgumentParser(description="Create a GIF and MP4 from Snake frames.")
    parser.add_argument("--clean", action="store_true", help="Delete frame_*.png files after creating the video")
    parser.add_argument("--label", type=str, default="", help="Label to add to output filename (e.g. --label neat => snake_game_neat.gif)")
    parser.add_argument("--dir", type=str, default=".", help="Directory to read frames from and write output to (default: current directory)")
    args = parser.parse_args()

    # Resolve target directory
    folder = Path(args.dir).resolve()

    if not folder.exists() or not folder.is_dir():
        print(f"Error: '{folder}' is not a valid directory.")
        return

    # Match all frame_XXXX.png files
    frame_files = sorted(
        [f for f in folder.glob("frame_*.png") if re.match(r"frame_\d{4}\.png", f.name)],
        key=lambda f: f.name
    )

    if not frame_files:
        print("No frames found.")
        return

    # Define label suffix
    suffix = f"_{args.label}" if args.label else ""

    # Output paths
    gif_path = folder / f"snake_game{suffix}.gif"
    mp4_path = folder / f"snake_game{suffix}.mp4"

    # Read frames
    images = [imageio.imread(f) for f in frame_files]

    # Save GIF
    imageio.mimsave(gif_path, images, duration=0.1)
    print(f"Saved GIF: {gif_path}")

    # Save MP4 (requires ffmpeg backend)
    try:
        imageio.mimsave(mp4_path, images, fps=10)
        print(f"Saved MP4: {mp4_path}")
    except Exception as e:
        print(f"Failed to save MP4: {e}")

    # Conditional cleanup
    if args.clean:
        for f in frame_files:
            f.unlink()
        print("Deleted all frame_*.png files.")

if __name__ == "__main__":
    main()
