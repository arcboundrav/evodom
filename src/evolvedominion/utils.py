import os
import pickle

from platformdirs import PlatformDirs
from evolvedominion.algorithm.models import SimpleModelSpace
from evolvedominion.params import (
    APPAUTHOR,
    APPNAME,
    MODELSPACE_CACHE_FILENAME,
)


_PLATFORMDIRS = PlatformDirs(appname=APPNAME, appauthor=APPAUTHOR)


def to_pkl(object, filepath):
    """ Serialize object. """
    with open(filepath, 'wb') as f:
        pickle.dump(object, f)


def from_pkl(filepath):
    """ Attempt to deserialize the object at filepath. """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj


class FileManager:
    """ Support careful OS-independent storage and retrieval of files. """
    def __init__(self, path, bad_directory_hint="File I/O defaulting to current working directory. {}"):
        self.path = path
        self.default_path = os.path.abspath(os.curdir)
        self.bad_directory_hint = bad_directory_hint
        self._try_to_create_directory()

    def _try_to_create_directory(self):
        if os.path.isdir(self.path):
            pass
        else:
            try:
                os.makedirs(self.path)
            except OSError as e:
                print(self.bad_directory_hint.format(e))

    def _try_to_load(self, filename):
        filepath = os.path.join(self.path, filename)
        try:
            return from_pkl(filepath)
        except OSError:
            filepath = os.path.join(self.default_path, filename)
            try:
                return from_pkl(filepath)
            except OSError as e:
                print("File: {} could not be loaded. {}".format(filename, e))
            else:
                print("File: {} loaded.".format(filename))
            finally:
                return None
        else:
            print("File: {} loaded.".format(filename))

    def _try_to_save(self, object, filename):
        filepath = os.path.join(self.path, filename)
        try:
            to_pkl(object, filepath)
        except OSError:
            filepath = os.path.join(self.default_path, filename)
            try:
                to_pkl(object, filepath)
            except OSError as e:
                print("File: {} could not be saved. {}".format(filename, e))
            else:
                print("File: {} saved.".format(filename))
        else:
            print("File: {} saved.".format(filename))

    def requires_overwrite(self, filename):
        filepath = os.path.join(self.path, filename)
        default_filepath = os.path.join(self.default_path, filename)
        if (not(os.path.exists(filepath)) and not(os.path.exists(default_filepath))):
            return False
        return True


class DataManager(FileManager):
    """ Wraps saving / loading Simulation data. """
    def __init__(self):
        super().__init__(path=_PLATFORMDIRS.user_data_dir,
                         bad_directory_hint="Data file I/O defaulting to current working directory. {}")

    def get_data_filename(self, simname):
        return "{}.evodom.data".format(simname)

    def save(self, simulation):
        self._try_to_save(simulation.data(), self.get_data_filename(simulation.simname))

    def load(self, simname):
        return self._try_to_load(self.get_data_filename(simname))


class CacheManager(FileManager):
    """ Wraps creating, caching, and loading models used by Simulations """
    def __init__(self):
        super().__init__(path=_PLATFORMDIRS.user_cache_dir,
                         bad_directory_hint="Cache file I/O will default to current working directory. {}")

    def load_cache(self):
        CACHE = self._try_to_load(MODELSPACE_CACHE_FILENAME)
        if (CACHE is None):
            MODELMAKER = SimpleModelSpace()
            CACHE = MODELMAKER.create_cache(models=MODELMAKER.get_full_model_space())
            self._try_to_save(CACHE)
        return CACHE


DATA_MANAGER = DataManager()
CACHE_MANAGER = CacheManager()
