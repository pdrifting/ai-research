## Atmospheric Entropy File Packing and ASCII Conversion Tool

### Overview:
This tool processes raw ASCII sampled atmospheric entropy files and converts them into equal fixed length blocks. It is designed so that researchers, students, and contributors with no programming background can use it to manipulate the data easily.

---

### What This Tool Does

Given a directory of raw ASCII sampled atmospheric entropy files, the tool will:

- Sort the files by modification time to maintain stream integrity
- Allows separate output directories for each output type
- Names output files using a flexible placeholder system
- Group them into batches (default: 8x 16384 Byte files per group)
- Combines each group into a single packed binary file
- Convert that new fix length file into binary ASCII **"0"**/**"1"** text
  - **NOTE**: Binary ASCII will consume a lot of disk space
- Save both outputs using user-defined filename formats
- Required to insert a counter placeholder [$C] in the filename
- Ensures deterministic, repeatable output

### Produces two parallel datasets:
- Packed ASCII fixed length files
- ASCII bitstreams for NIST verification (1MB minimum required for NIST examination)

---

2. Basic Usage

To run the tool on a directory of files:
<pre>
python pack_entropy.py ./input_directory
</pre>

This will:
- Group files in batches of 8
- Write both output types into the same directory
- Use default filename formats

---

3. Filename Placeholder: [$C]

All output filenames must contain the placeholder:
<pre>
[$C]
</pre>

This placeholder is replaced with a zero-padded counter.

Example:
<pre>
packed_block_[$C].bin
</pre>
Becomes:
<pre>
packed_block_0000.bin
packed_block_0001.bin
packed_block_0002.bin
</pre>

Padding width is controlled by **--cw** or **--cw-auto**.

---

4. Output Directory Options

Directory for packed binary files:
<pre>
--out-pack "directory"
</pre>

Directory for ASCII binary files:
<pre>
--out-bin "directory"
</pre>
  
If not provided, both default to the input directory.
The output directory if different must exist already.

---

5. Filename Format Options

Packed binary filename format:
<pre>
--pfmt "packed_block_[$C].bin"
</pre>

ASCII binary filename format:
<pre>
--bfmt "ascii_bits_[$C].txt"
</pre>

Both formats must contain the placeholder [$C].

---

6. Counter Width Options

Automatic width (default):
<pre>
--cw-auto
</pre>

Automatically sizes the counter width based on the number of groups.

Example:
If there are 1,234 groups → width = 4 → 0000 to 1233

Manual width:
<pre>
--cw 6
</pre>

Forces a fixed width:
<pre>
000000, 000001, 000002, ...
</pre>

Useful for large datasets or consistent naming across runs.

---

7. Group Size

Default group size is 8 (minimum for NIST tests):

<pre>
--group-size 8
</pre>

You may change it:

<pre>
--group-size 16
</pre>

---

8. Full Example

<pre>
python atmospher_packer.py ./samples \
    --group-size 8 \
    --out-pack ./packed \
    --out-bin ./ascii \
    --pfmt "packed_block_[$C].bin" \
    --bfmt "ascii_bits_[$C].txt" \
    --cw-auto
</pre>

This will:
- Read files from ./samples
- Group them in batches of 8
- Write new ASCII fixed length files to ./packed
- Write ASCII bitstreams to ./ascii
  - **NOTE**: Output directory will be created if it doesn't exist
- Use automatic counter width
- Produce filenames like:
<pre>
  packed_block_0000.bin
  ascii_bits_0000.txt
</pre>

---

9. Requirements

- Python 3.7 or newer
- No external libraries required

---

10. Notes for Researchers and Students

- You do not need to understand Python to use this tool
- All configuration is done through command-line options
- The script is heavily commented for educational purposes
- The tool is deterministic: same input to same output
- ASCII output is suggested for long term storage.
- Binary output is large because each byte becomes 8 characters.

---

11. File Overview

<pre>
atmospheric_parser.py   (main tool)
readme.md               (this documentation)
</pre>
