import os
import argparse


# ------------------------------------------------------------
# Function: pack_and_convert
# Purpose:
#   - Read files from a directory
#   - Group them into fixed-size batches
#   - Combine each group into one binary file
#   - Convert that binary data into ASCII '0'/'1' text
#   - Save both outputs using user-defined naming patterns
#   - Save outputs into user-defined directories
# ------------------------------------------------------------
def pack_and_convert(directory, group_size, packed_fmt, binary_fmt,
                     out_pack, out_bin, counter_width, auto_width):

    # Convert directory path to an absolute path for safety
    directory = os.path.abspath(directory)

    # If output directories were not provided, default to input directory
    if out_pack is None:
        out_pack = directory
    if out_bin is None:
        out_bin = directory

    # Ensure output directories exist
    os.makedirs(out_pack, exist_ok=True)
    os.makedirs(out_bin, exist_ok=True)

    # ------------------------------------------------------------
    # STEP 1: Collect all files in the directory
    # ------------------------------------------------------------
    files = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]

    # ------------------------------------------------------------
    # STEP 2: Sort files by modification time (oldest → newest)
    # ------------------------------------------------------------
    files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)))

    # ------------------------------------------------------------
    # STEP 3: Determine number of groups
    # ------------------------------------------------------------
    total_groups = len(files) // group_size

    # ------------------------------------------------------------
    # STEP 4: Determine counter width
    # ------------------------------------------------------------
    # If user provided --cw, use that.
    # If user provided --cw-auto, compute width from number of groups.
    # If neither is provided, default to auto-width.
    if counter_width is not None:
        pad = counter_width
    else:
        # Auto-width mode
        pad = len(str(total_groups)) if total_groups > 0 else 1

    # ------------------------------------------------------------
    # STEP 5: Validate filename formats
    # ------------------------------------------------------------
    if "[$C]" not in packed_fmt or "[$C]" not in binary_fmt:
        raise ValueError("Both --pfmt and --bfmt must contain the placeholder '[$C]'.")

    # ------------------------------------------------------------
    # STEP 6: Process files in chunks
    # ------------------------------------------------------------
    for idx in range(0, len(files), group_size):

        group = files[idx:idx + group_size]

        # Skip incomplete groups
        if len(group) < group_size:
            print(f"Skipping incomplete group at index {idx}")
            continue

        # Determine group number (0-based)
        block_index = idx // group_size

        # Convert index to zero-padded string
        block_str = str(block_index).zfill(pad)

        # ------------------------------------------------------------
        # STEP 7: Build output filenames using user formats
        # ------------------------------------------------------------
        packed_name = packed_fmt.replace("[$C]", block_str)
        binary_name = binary_fmt.replace("[$C]", block_str)

        packed_path = os.path.join(out_pack, packed_name)
        binary_path = os.path.join(out_bin, binary_name)

        # ------------------------------------------------------------
        # STEP 8: Read and combine all files in the group
        # ------------------------------------------------------------
        data = b""
        for fname in group:
            with open(os.path.join(directory, fname), "rb") as f:
                data += f.read()

        # ------------------------------------------------------------
        # STEP 9: Write the combined binary file
        # ------------------------------------------------------------
        with open(packed_path, "wb") as f:
            f.write(data)

        # ------------------------------------------------------------
        # STEP 10: Convert binary data to ASCII '0'/'1' text
        # ------------------------------------------------------------
        binary_str = ''.join(f"{byte:08b}" for byte in data)

        # ------------------------------------------------------------
        # STEP 11: Write ASCII-binary file
        # ------------------------------------------------------------
        with open(binary_path, "w") as f:
            f.write(binary_str)

        print(f"Created: {packed_name}  AND  {binary_name}")

    print("\nDone. All groups processed.")


# ------------------------------------------------------------
# main()
# Sets up the command-line interface for researchers.
# ------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(
        description="Group entropy files and produce binary + ASCII-bit outputs."
    )

    # Required: directory containing input files
    parser.add_argument(
        "directory",
        help="Directory containing the input files."
    )

    # Optional: number of files per group
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Number of files per group (default: 8)."
    )

    # Optional: output directory for packed binary files
    parser.add_argument(
        "--out-pack",
        type=str,
        default=None,
        help="Directory for packed binary output files (default: same as input)."
    )

    # Ensure output directories exist; create them if missing
    if not os.path.exists(out_pack):
        os.makedirs(out_pack)
        print(f"Output directory missing, created: {out_pack}")

    # Optional: output directory for ASCII-binary files
    parser.add_argument(
        "--out-bin",
        type=str,
        default=None,
        help="Directory for ASCII-binary output files (default: same as input)."
    )

    # Optional: naming format for packed binary files
    parser.add_argument(
        "--pfmt",
        type=str,
        default="packed_block_[$C].bin",
        help="Filename format for packed binary files. Must contain '[$C]'."
    )

    # Optional: naming format for ASCII-binary files
    parser.add_argument(
        "--bfmt",
        type=str,
        default="ascii_bits_[$C].txt",
        help="Filename format for ASCII-binary files. Must contain '[$C]'."
    )

    # Optional: manual counter width
    parser.add_argument(
        "--cw",
        type=int,
        default=None,
        help="Force a specific counter width (e.g., 6 → 000000)."
    )

    # Optional: auto counter width (based on dataset size)
    parser.add_argument(
        "--cw-auto",
        action="store_true",
        help="Automatically size the counter width based on number of groups."
    )

    args = parser.parse_args()

    pack_and_convert(
        args.directory,
        args.group_size,
        args.pfmt,
        args.bfmt,
        args.out_pack,
        args.out_bin,
        args.cw,
        args.cw_auto
    )


# Standard Python entry point
if __name__ == "__main__":
    main()
