import subprocess
from pathlib import Path
import time


def pdf_to_mmd(path_input: str):
    """
    Convert a PDF file to MMD format using the Nougat library
    https://github.com/facebookresearch/nougat

    stream stderr to the front end
    """
    output_dir = "../documents/mmds"
    command = ['nougat', path_input, "-o", output_dir]
    subprocess.run(command)
    time.sleep(1)
    # Change the math delimiter to the common delimiter used in MMD
    with open(f"{output_dir}/{str(Path(path_input).stem)}.mmd", "r+") as doc:
        content = doc.read()
        print(content)

        content = content.replace(r"\[", "$$").replace(r"\]", "$$")
        content = content.replace(r"\(", "$").replace(r"\)", "$")
        # delete the content of the file
        doc.seek(0)
        doc.truncate()
        doc.write(content)



