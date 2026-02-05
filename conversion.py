import csv

# Configuration
input_filename = 'new.txt'
output_filename = 'hpodata_2026_1-2.csv'

# Header columns based on file description
header = ["YYYY", "MM", "DD", "hh.h", "hh._m", "days", "days_m", "Hp60", "ap60", "D"]

def convert_to_csv(in_file, out_file):
    try:
        with open(in_file, 'r') as infile, open(out_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            
            # Write the CSV Header
            writer.writerow(header)
            
            for line in infile:
                # Skip comment lines (headers)
                if line.startswith('#'):
                    continue
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Split line by whitespace
                parts = line.split()
                
                # Ensure the line has enough data columns (expecting 10)
                if len(parts) >= 10:
                    writer.writerow(parts[:10])
                    
        print(f"Successfully converted '{in_file}' to '{out_file}'")

    except FileNotFoundError:
        print(f"Error: The file '{in_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    convert_to_csv(input_filename, output_filename)