import os
import sys

def rename_entropy_files(directory):
    # Normalize path
    directory = os.path.abspath(directory)

    # Get all files (ignore directories)
    files = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]

    # Sort by modification time (oldest first)
    files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)))

    # Determine zero-padding width
    pad = len(str(len(files)))

    for idx, filename in enumerate(files, start=1):
        old_path = os.path.join(directory, filename)
        new_name = f"atmospheric.{str(idx).zfill(pad)}.ascii_byte_encoded.txt"
        new_path = os.path.join(directory, new_name)

        print(f"Renaming: {filename} -> {new_name}")
        os.rename(old_path, new_path)

    print("\nDone. All files renamed in chronological order.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_entropy_files.py <directory>")
        sys.exit(1)

    rename_entropy_files(sys.argv[1])
