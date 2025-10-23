import os


def write_to_file(str, filename):
    # if the directory where the filename does not exist, create it
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, "w") as f:
        f.write(str)
        f.close()
