import os

def split_bits(input_file, num_files=10, bits_per_file=1000000):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r') as f:
        # We read as needed to handle large files efficiently
        for i in range(num_files):
            chunk = f.read(bits_per_file)
            if not chunk:
                break
            
            output_name = f"bits_part_{i+1}.txt"
            with open(output_name, 'w') as out:
                out.write(chunk)
            print(f"Created {output_name} ({len(chunk)} bits)")

if __name__ == "__main__":
    split_bits("generated_bits.txt")