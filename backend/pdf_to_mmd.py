import subprocess


def pdf_to_mmd(path_input: str):
    """
    Convert a PDF file to MMD format using the Nougat library
    https://github.com/facebookresearch/nougat

    stream stderr to the front end
    """
    output_dir = "../documents/mmds"
    command = ['nougat', path_input, "-o", output_dir]
    subprocess.run(command)



