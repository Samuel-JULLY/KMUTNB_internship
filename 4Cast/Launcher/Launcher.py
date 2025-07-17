import subprocess
import os

# Full path to your .bat file
# Make sure the path is correct.
# If the .bat is in the same folder as your Python script,
# you can simply use its filename.
bat_file_path = "Launcher.bat" 

# Check if the .bat file exists
if not os.path.exists(bat_file_path):
    print(f"Error: The file '{bat_file_path}' was not found.")
else:
    try:
        # Execute the .bat file
        # shell=True is necessary to run .bat files on Windows
        # capture_output=True captures stdout and stderr (Python 3.7+)
        # text=True decodes output as text
        print(f"Launching {bat_file_path}...")
        result = subprocess.run(bat_file_path, shell=True, capture_output=True, text=True, check=True)
        
        print("\n--- .bat script output ---")
        print(result.stdout)
        
        if result.stderr:
            print("\n--- .bat script errors ---")
            print(result.stderr)

        print("\n--- .bat script finished ---")

    except subprocess.CalledProcessError as e:
        print(f"The .bat script returned an error: {e.returncode}")
        print(f"Standard output: {e.stdout}")
        print(f"Error output: {e.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Optional: Pause to see the output if the EXE closes quickly
# input("Press Enter to exit...")