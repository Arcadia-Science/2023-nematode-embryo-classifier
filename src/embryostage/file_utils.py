import pathlib


def find_repo_root(current_dirpath):
    '''
    find the absolute path of the nearest local git repo
    '''
    current_dirpath = pathlib.Path(current_dirpath).absolute()
    while True:
        if (current_dirpath / ".git").exists():
            return current_dirpath

        # Move up one level in the directory hierarchy
        current_dirpath = current_dirpath.parent

        # If we've reached the root of the filesystem, raise an error
        if current_dirpath == current_dirpath.parent:
            raise FileNotFoundError("Could not find the root of the local git repo.")
