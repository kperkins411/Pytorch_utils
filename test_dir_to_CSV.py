import shutil, tempfile
import pandas as pd
from unittest import TestCase, main
from dir_to_CSV import directories_to_CSV

class TestCaseBase(TestCase):
    '''
    creates all the temp files needed
    '''
    # numb_files = [5,4,3,2]
    # numb_dirs = len(numb_files)

    def setUp(self):
        #dont use this class directly
        raise NotImplemented

    def do_setup(self):
        self.numb_dirs = len(self.number_files_per_directory)
        self.dir_list = []
        self.file_list = []
        # Create a temporary directory
        self.root_dir = tempfile.mkdtemp()
        # create subdirs
        for _ in range(self.numb_dirs):
            self.dir_list.append(tempfile.mkdtemp(dir=self.root_dir))
        for numb, dir in enumerate(self.dir_list):
            for val in range(self.number_files_per_directory[numb]):
                self.file_list.append((tempfile.mkstemp(suffix=".png", dir=dir))[1])

        self.csv_file = "tmp.csv"
        dirs = []
        for dir in self.dir_list:
            dirs.append(dir.split('/')[-1])
        directories_to_CSV(csv_file=self.csv_file, subdirs=dirs, root_dir=self.root_dir,
                           df_column_names=["file", "label"], use_full_path=False)
        #read into a dataframe
        self.df = pd.read_csv(self.csv_file)

    def tearDown(self):
        # Removedirs the directory after the test
        shutil.rmtree(self.root_dir)


class TestSingleDir(TestCaseBase):
    def setUp(self):
        self.number_files_per_directory = [5]
        self.do_setup()

    def test_correct_number_files(self):
        #ensure same number of records
        self.assertEqual(self.df.shape[0], len(self.file_list))

    def test_correct_files(self):
        #see whats in them
        for index, row in self.df.iterrows():
            self.assertTrue(row[0] in self.file_list)

class TestMultipleSubDir(TestCaseBase):
    def setUp(self):
        self.number_files_per_directory = [5,4,3,2]
        self.do_setup()

    def test_correct_number_files(self):
        # ensure same number of records
        self.assertEqual(self.df.shape[0], len(self.file_list))

    def test_correct_files(self):
        # see whats in them
        for index, row in self.df.iterrows():
            self.assertTrue(row[0] in self.file_list)
