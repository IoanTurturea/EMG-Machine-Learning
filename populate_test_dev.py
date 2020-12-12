import os, shutil, random

# code, may be omitted or run every time
# it populates dev, test folders
# with files chosen randomly from Train
# before adding files to them
# it deletes all the content

# obs: for unknown reasons the
# folders are populated with slightly
# different number of files (eg: 113 and 110)

path = "/home/ioan/Desktop/Database/"

# make proportions, for dev/test
# I took 10% for instance
procent = 0.1
number_of_files = len(os.listdir(path + "Train"))
files_procent = int(number_of_files * procent)

# clean the Test/Dev folders by deleting them
# and recreate them
shutil.rmtree(path + "Test")
os.makedirs(path + "Test")
shutil.rmtree(path + "Val")
os.makedirs(path + "Val")

for i in range(0, files_procent):
    chosen_file_name_for_test = random.choice(os.listdir(path + "Train"))
    chosen_file_name_for_val = random.choice(os.listdir(path + "Train"))
    # shutil.copy(source_path, dest_path)
    shutil.copy(path + "Train/" + chosen_file_name_for_test, path + "Test")
    shutil.copy(path + "Train/" + chosen_file_name_for_val, path + "Val")

# end of making Test/Val