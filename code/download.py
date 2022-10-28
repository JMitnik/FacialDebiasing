# Dependencies
import os
import urllib.request
import tarfile
import pandas as pd

# Set path to current directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def ensure_path(path):
    """Makes sure a path exists and otherwise creates one."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    return path


# Fetch train data
path_to_train_data = 'data/h5_train/train_face.h5'
train_url = 'https://www.dropbox.com/s/l5iqduhe0gwxumq/train_face.h5?dl=1'

if not os.path.exists(path_to_train_data):
    print(f"Going to fetch training data, will place it in {path_to_train_data}.")
    urllib.request.urlretrieve(train_url, ensure_path(path_to_train_data))
else:
    print("Train data already exists.")


# Fetch train data
path_to_test_data = 'data/ppb/PPB.tar'
test_url = 'https://www.dropbox.com/s/l0lp6qxeplumouf/PPB.tar?dl=1'

if not os.path.exists(path_to_test_data):
    print(f"Going to fetch test data, will place it in {path_to_test_data}.")
    urllib.request.urlretrieve(test_url, ensure_path(path_to_test_data))

    with tarfile.open(f"{path_to_test_data}", "r:") as tar:
        members = [tar_mem for tar_mem in tar.getmembers()]
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, os.path.dirname(path_to_test_data), members=members)
else:
    print("Test data already exists.")

# Change the bi.fitz column name
path_to_ppb_metadata = 'data/ppb/PPB-2017/PPB-2017-metadata.csv'
pd.read_csv(path_to_ppb_metadata).rename(columns={'bi.fitz': 'bi_fitz'}).to_csv(path_to_ppb_metadata, index=False)
