import glob
import os


def delete_files(file_path: list) -> None:
    """
    Remove batch files
    :param file_path: list of path for each batch file to remove
    :return: None
    """
    for file in file_path:
        os.remove(file)


def move_files(file_path: list) -> None:
    """
    Move files from subdir and delete the subdir
    :param file_path: list of path for each file
    :return: None
    """
    for file in file_path:
        os.rename(file, os.path.join(os.path.dirname(file), "../..", os.path.basename(file)))
        os.rmdir(os.path.dirname(file))


batch_path = glob.glob(os.path.join("../data", "yahoo_movies", "scores", "eps_*", "*", "batch*.pk"))
seed_path = glob.glob(os.path.join("../data", "yahoo_movies", "scores", "eps_*", "*", "seed*.pk"))

if __name__ == '__main__':
    move_files(batch_path)
    move_files(seed_path)

