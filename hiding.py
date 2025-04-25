import subprocess
import os


def embed_with_openstego(message: str, clean_image_path: str, stego_image_path: str,
                         message_file: str = "temp_message.txt", algorithm: str = "RandomLSB"):
    """
    Embed a message into a clean image using OpenStego's RandomLSB algorithm.

    Args:
        message (str): The message to embed (e.g., "Test123").
        clean_image_path (str): Path to the clean BMP image.
        stego_image_path (str): Path where the stego image will be saved.
        message_file (str): Temporary file to store the message.
        algorithm (str): OpenStego algorithm to use (default: RandomLSB).

    Returns:
        bool: True if embedding succeeded, False otherwise.
    """
    try:
        # Write the message to a temporary file
        with open(message_file, 'w') as f:
            f.write(message)

        # Construct the OpenStego command
        command = [
            "openstego",
            "embed",
            "-a", algorithm,
            "-mf", message_file,
            "-cf", clean_image_path,
            "-sf", stego_image_path
        ]

        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"OpenStego embedding output: {result.stdout}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error during OpenStego embedding: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    finally:
        # Clean up the temporary message file
        if os.path.exists(message_file):
            os.remove(message_file)


# Example usage
if __name__ == "__main__":
    # Embed "Test123" into test_stego_new.bmp
    success = embed_with_openstego(
        message="Test123",
        clean_image_path="openstego/original-bmp/image1.bmp",
        stego_image_path="test_stego_new.bmp"
    )
    if success:
        print("Successfully embedded 'Test123' into test_stego_new.bmp")

    # Embed "StegoTest123" into image1.bmp
    success = embed_with_openstego(
        message="StegoTest123",
        clean_image_path="openstego/original-bmp/image1.bmp",
        stego_image_path="openstego/encrypted-from-bmp/image1.bmp"
    )
    if success:
        print("Successfully embedded 'StegoTest123' into openstego/encrypted-from-bmp/image1.bmp")