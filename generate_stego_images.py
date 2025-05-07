"""
openstego download: https://github.com/syvaidya/openstego/releases  (version 0.8.61)
sudo apt install outguess  # version 0.4
sudo apt install steghide

Facts:
1. OpenStego could embed  to bmp, png lossless formats - jpg files will be converted to bmp
After this operation, 225 kB of additional disk space will be used.

OpenStego is a steganography application that provides two functionalities:
  1. Data Hiding: It can hide any data within an image file.
  2. Watermarking: Watermarking image files with an invisible signature. It
     can be used to detect unauthorized file copying.

For GUI:
  java -jar <path>/openstego.jar

For command line interface:
  java -jar <path>/openstego.jar <command> [options]

The first argument must be a command. Valid commands are:

  embed, --embed                Embed message into cover file
  extract, --extract            Extract message from stego file
  gensig, --gensig              Generate a signature for watermarking
  embedmark, --embedmark        Watermark a cover file using signature
  checkmark, --checkmark        Checks for watermark presence in the file
  algorithms, --algorithms      List down supported steganography algorithms
  readformats --readformats     List down supported formats for cover file
  writeformats, --writeformats  List down supported formats for stego file
  help, --help                  Display this help. To get help on options
                                specific to an algorithm, provide the name of
                                the algorithm using '-a' option

Following option is common for all commands other than 'algorithms':

    -a, --algorithm <algorithm_name>
        Name of the steganography algorithm to use. List of the supported
        algorithms can be retrieved using 'algorithms' command

'embed' options:

    -mf, --messagefile <filename>
        Source message/data file. If this option is not provided or '-'
        (without quotes) is provided as the value then the message data is
        read from stdin

    -cf, --coverfile <filename>
        Cover file in which the message will be embedded. This option supports
        '*' and '?' wildcards for filename. If wildcard filename is provided
        then make sure that it is surrounded by double quotes.
        Multiple filenames can also be provided by separating them with ';'
        (semi-colon).
        If the wildcard expression returns more than one file, then '-sf'
        option is ignored, and each coverfile is overwritten with the
        corresponding generated stegofile

    -sf, --stegofile <filename>
        Output stego file containing the embedded message. If this option is
        not provided or '-' (without quotes) is provided as the value then the
        stego file is written to stdout

    -c, --compress
        Compress the message file before embedding (default)

    -C, --nocompress
        Do not compress the message file before embedding

    -e, --encrypt
        Encrypt the message file before embedding

    -E, --noencrypt
        Do not encrypt the message file before embedding (default)

    -p, --password <password>
        Password to be used for encryption. If this is not provided then
        prompt will be displayed for entry

    -A, --cryptalgo <crypto algorithm>
        Algorithm to be used for encryption. Defaults to AES128. Possible
        values are AES128 or AES256. Please note that AES256 will only be
        supported with JRE having unlimited strength jurisdiction policy

'extract' options:

    -sf, --stegofile <filename>
        Stego file containing the embedded message

    -xf, --extractfile <filename>
        Optional filename for the extracted data. Use this to override the
        filename embedded in the stego file

    -xd, --extractdir <dir>
        Directory where the message file will be extracted. If this option is
        not provided, then the file is extracted to current directory

    -p, --password <password>
        Password to be used for decryption. If this is not provided then
        prompt will be displayed for entry (if the message is encrypted)

'gensig' options:

    -gf, --sigfile <filename>
        Output signature file that can be used to watermark files. If this
        option is not provided or '-' (without quotes) is provided as the
        value then the signature file is written to stdout

    -p, --password <password>
        Password to be used for generation of signature. If this is not
        provided then prompt will be displayed for entry

'embedmark' options:

    -gf, --sigfile <filename>
        Watermarking signature file. If this option is not provided or '-'
        (without quotes) is provided as the value then the signature data is
        read from stdin

    -cf, --coverfile <filename>
        Cover file which needs to be digitally watermarked. This option supports
        '*' and '?' wildcards for filename. If wildcard filename is provided
        then make sure that it is surrounded by double quotes.
        Multiple filenames can also be provided by separating them with ';'
        (semi-colon).
        If the wildcard expression returns more than one file, then '-sf'
        option is ignored, and each coverfile is overwritten with the
        corresponding generated stegofile

    -sf, --stegofile <filename>
        Output stego file containing the embedded watermark. If this option is
        not provided or '-' (without quotes) is provided as the value then the
        stego file is written to stdout

'checkmark' options:

    -sf, --stegofile <filename>
        Stego file containing the embedded watermark

    -gf, --sigfile <filename>
        Signature file which was used to watermark the file

Examples:

  To embed secret.txt into wallpaper.png and generate the output into test.png:

      java -jar <path>/openstego.jar embed -a lsb -mf secret.txt \
        -cf wallpaper.png -sf test.png
   OR
      java -jar <path>/openstego.jar --embed --algorithm=lsb \
        --messagefile=secret.txt --coverfile=wallpaper.png --stegofile=test.png

  To extract embedded data from test.png to /data directory:

      java -jar <path>/openstego.jar extract -a lsb -sf test.png -xd /data
   OR
      java -jar <path>/openstego.jar extract --algorithm=lsb \
        --stegofile=test.png --extractdir=/data

  To generate a signature file:

      java -jar <path>/openstego.jar gensig -a dwtxie -gf my.sig
   OR
      java -jar <path>/openstego.jar --gensig --algorithm=dwtxie \
        --sigfile=my.sig

  To embed signature into owned.png and generate the output into out.png:

      java -jar <path>/openstego.jar embedmark -a dwtxie -gf my.sig \
        -cf owned.png -sf out.png
   OR
      java -jar <path>/openstego.jar --embedmark --algorithm=dwtxie \
        --sigfile=my.sig --coverfile=owned.png --stegofile=out.png

  To check for watermark in test.png using my.sig signature file:

      java -jar <path>/openstego.jar checkmark -a dwtxie -gf my.sig \
        -sf test.png
   OR
      java -jar <path>/openstego.jar checkmark --algorithm=dwtxie \
        --sigfile=my.sig --stegofile=test.png

  Piping example:

      ls -R | java -jar <path>/openstego.jar embed -a lsb > test.png

  Wildcard example (Please note that the double quotes are important):

      java -jar <path>/openstego.jar embed -a lsb \
        -cf "img???.png;wall*.png" -mf watermark.txt

-------------------------------------------------------------------------------------
2. Outguess:
outguess [options] [<input file> [<output file>]]
        -h           print this usage help text and exit
        -[sS] <n>    iteration start, capital letter for 2nd dataset
        -[iI] <n>    iteration limit
        -[kK] <key>  key
        -[dD] <name> filename of dataset
        -[eE]        use error correcting encoding
        -p <param>   parameter passed to destination data handler
        -r           retrieve message from data
        -x <n>       number of key derivations to be tried
        -m           mark pixels that have been modified
        -t           collect statistic information
        -F[+-]       turns statistical steganalysis foiling on/off.
                     The default is on.

-------------------------------------------------------------------------------------
3. StegHide:

steghide version 0.5.1

the first argument must be one of the following:
 embed, --embed          embed data
 extract, --extract      extract data
 info, --info            display information about a cover- or stego-file
   info <filename>       display information about <filename>
 encinfo, --encinfo      display a list of supported encryption algorithms
 version, --version      display version information
 license, --license      display steghide's license
 help, --help            display this usage information

embedding options:
 -ef, --embedfile        select file to be embedded
   -ef <filename>        embed the file <filename>
 -cf, --coverfile        select cover-file
   -cf <filename>        embed into the file <filename>
 -p, --passphrase        specify passphrase
   -p <passphrase>       use <passphrase> to embed data
 -sf, --stegofile        select stego file
   -sf <filename>        write result to <filename> instead of cover-file
 -e, --encryption        select encryption parameters
   -e <a>[<m>]|<m>[<a>]  specify an encryption algorithm and/or mode
   -e none               do not encrypt data before embedding
 -z, --compress          compress data before embedding (default)
   -z <l>                 using level <l> (1 best speed...9 best compression)
 -Z, --dontcompress      do not compress data before embedding
 -K, --nochecksum        do not embed crc32 checksum of embedded data
 -N, --dontembedname     do not embed the name of the original file
 -f, --force             overwrite existing files
 -q, --quiet             suppress information messages
 -v, --verbose           display detailed information

extracting options:
 -sf, --stegofile        select stego file
   -sf <filename>        extract data from <filename>
 -p, --passphrase        specify passphrase
   -p <passphrase>       use <passphrase> to extract data
 -xf, --extractfile      select file name for extracted data
   -xf <filename>        write the extracted data to <filename>
 -f, --force             overwrite existing files
 -q, --quiet             suppress information messages
 -v, --verbose           display detailed information

options for the info command:
 -p, --passphrase        specify passphrase
   -p <passphrase>       use <passphrase> to get info about embedded data

To embed emb.txt in cvr.jpg: steghide embed -cf cvr.jpg -ef emb.txt
To extract embedded data from stg.jpg: steghide extract -sf stg.jpg

 steghide --encinfo
encryption algorithms:
<algorithm>: <supported modes>...
cast-128: cbc cfb ctr ecb ncfb nofb ofb
gost: cbc cfb ctr ecb ncfb nofb ofb
rijndael-128: cbc cfb ctr ecb ncfb nofb ofb
twofish: cbc cfb ctr ecb ncfb nofb ofb
arcfour: stream
cast-256: cbc cfb ctr ecb ncfb nofb ofb
loki97: cbc cfb ctr ecb ncfb nofb ofb
rijndael-192: cbc cfb ctr ecb ncfb nofb ofb
saferplus: cbc cfb ctr ecb ncfb nofb ofb
wake: stream
des: cbc cfb ctr ecb ncfb nofb ofb
rijndael-256: cbc cfb ctr ecb ncfb nofb ofb
serpent: cbc cfb ctr ecb ncfb nofb ofb
xtea: cbc cfb ctr ecb ncfb nofb ofb
blowfish: cbc cfb ctr ecb ncfb nofb ofb
enigma: stream
rc2: cbc cfb ctr ecb ncfb nofb ofb
tripledes: cbc cfb ctr ecb ncfb nofb ofb

-------------------------------------------------------------------------------------

Paths:
./training  # training root
./training/embeds  # embedded secrets, testfile.txt (contains "Secret message!")
./training/originals  # bmp, png and jpg images, variable sizes, variable tools - some contains EXIF informatin
./training/openstego # bmp or png lossless files, if original was jpg, resulting .bmp
./training/steghide # JPEG, BMP, WAV and AU file formats
./training/outguess # Netpbm and JPEG file formats

"""
import os
import subprocess
from PIL import Image
import pandas as pd

# Paths and configuration
ORIGINAL_DIR = "./training/originals/"
STEGO_OPENSTEGA_DIR = "./training/openstego/"
STEGO_STEGHIDE_DIR = "./training/steghide/"
STEGO_OUTGUESS_DIR = "./training/outguess/"
SECRET_MESSAGE_FILE = "./training/embeds/testfile.txt"
PASSPHRASE = "set"
STEGO_METADATA_FILE = "./training/stego_metadata.csv"
VERIFY_MESSAGE_PATH = "./training/extracted/verify.txt"

# Create output directories if missing
os.makedirs(STEGO_OPENSTEGA_DIR, exist_ok=True)
os.makedirs(STEGO_STEGHIDE_DIR, exist_ok=True)
os.makedirs(STEGO_OUTGUESS_DIR, exist_ok=True)
os.makedirs("./training/extracted/", exist_ok=True)

# Supported input/output formats for each tool
filetype_matrix = {
    "openstego": {
        "input": ["bmp", "png", "jpg"],
        "output": {"bmp": "bmp", "png": "png", "jpg": "bmp"}
    },
    "steghide": {
        "input": ["bmp", "jpg", "wav", "au"],
        "output": {"bmp": "bmp", "jpg": "jpg", "wav": "wav", "au": "au"}
    },
    "outguess": {
        "input": ["jpg", "ppm", "pnm", "pgm", "pbm"],
        "output": {"jpg": "jpg", "ppm": "ppm", "pnm": "pnm", "pgm": "pgm", "pbm": "pbm"}
    }
}

# Convert BMP to JPEG for outguess compatibility

def convert_to_jpeg(input_path, output_path):
    img = Image.open(input_path).convert("RGB")
    img.save(output_path, "JPEG", quality=95)

# Attempt to extract and verify message from stego image

def verify_extraction(tool, stego_path):
    try:
        if tool == "steghide":
            cmd = ["steghide", "extract", "-sf", stego_path, "-xf", VERIFY_MESSAGE_PATH, "-p", PASSPHRASE, "-f"]
        elif tool == "openstego":
            cmd = ["openstego", "extract", "-sf", stego_path, "-xf", VERIFY_MESSAGE_PATH, "-p", PASSPHRASE]
        elif tool == "outguess":
            cmd = ["outguess", "-r", stego_path, VERIFY_MESSAGE_PATH]
        else:
            return False

        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open(VERIFY_MESSAGE_PATH, "r") as f:
            content = f.read().strip()
        return content == "Secret message!"
    except Exception:
        return False

# Process each image in ORIGINAL_DIR
metadata_records = []
original_files = sorted([f for f in os.listdir(ORIGINAL_DIR) if os.path.isfile(os.path.join(ORIGINAL_DIR, f))])

for idx, original_file in enumerate(original_files, 1):
    original_path = os.path.join(ORIGINAL_DIR, original_file)
    base_name, ext = os.path.splitext(original_file)
    ext = ext[1:].lower()

    # --- Steghide ---
    if ext in filetype_matrix["steghide"]["input"]:
        output_path = os.path.join(STEGO_STEGHIDE_DIR, f"{base_name}_steghide.{ext}")
        steghide_cmd = [
            "steghide", "embed",
            "-cf", original_path,
            "-ef", SECRET_MESSAGE_FILE,
            "-sf", output_path,
            "-p", PASSPHRASE,
            "-f"
        ]
        try:
            subprocess.run(steghide_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if verify_extraction("steghide", output_path):
                print(f"[{idx}] Steghide: {output_path}")
                metadata_records.append({"original": original_file, "tool": "steghide", "outfile": os.path.basename(output_path)})
        except subprocess.CalledProcessError as e:
            print(f"[Steghide Error] {original_file}: {e.stderr.decode()}")

    # --- OutGuess ---
    if ext not in filetype_matrix["outguess"]["input"] and ext == "bmp":
        temp_jpeg = os.path.join(STEGO_OUTGUESS_DIR, f"{base_name}_temp.jpg")
        convert_to_jpeg(original_path, temp_jpeg)
        cover_path = temp_jpeg
    elif ext in filetype_matrix["outguess"]["input"]:
        cover_path = original_path
    else:
        cover_path = None

    if cover_path:
        outguess_out = os.path.join(STEGO_OUTGUESS_DIR, f"{base_name}_outguess.jpg")
        outguess_cmd = ["outguess", "-d", SECRET_MESSAGE_FILE, cover_path, outguess_out]
        try:
            subprocess.run(outguess_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if verify_extraction("outguess", outguess_out):
                print(f"[{idx}] OutGuess: {outguess_out}")
                metadata_records.append({"original": original_file, "tool": "outguess", "outfile": os.path.basename(outguess_out)})
        except subprocess.CalledProcessError as e:
            print(f"[OutGuess Error] {original_file}: {e.stderr.decode()}")
        if cover_path.endswith("_temp.jpg") and os.path.exists(cover_path):
            os.remove(cover_path)

    # --- OpenStego ---
    if ext in filetype_matrix["openstego"]["input"]:
        result_ext = filetype_matrix["openstego"]["output"][ext]
        output_path = os.path.join(STEGO_OPENSTEGA_DIR, f"{base_name}_openstego.{result_ext}")
        openstego_cmd = [
            "openstego", "embed",
            "-mf", SECRET_MESSAGE_FILE,
            "-cf", original_path,
            "-sf", output_path,
            "-p", PASSPHRASE
        ]
        try:
            subprocess.run(openstego_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if verify_extraction("openstego", output_path):
                print(f"[{idx}] OpenStego: {output_path}")
                metadata_records.append({"original": original_file, "tool": "openstego", "outfile": os.path.basename(output_path)})
        except subprocess.CalledProcessError as e:
            print(f"[OpenStego Error] {original_file}: {e.stderr.decode()}")

# Save metadata to CSV
if metadata_records:
    df = pd.DataFrame(metadata_records)
    df.to_csv(STEGO_METADATA_FILE, index=False)
    print(f"Metadata saved to {STEGO_METADATA_FILE}")

print("All stego image generation complete.")
