# Dependencies
import os
import urllib.request
import tarfile

def ensure_path(path):
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
path_to_test_data = 'data/pbb/PBB.tar'
test_url = 'https://www.dropbox.com/s/l0lp6qxeplumouf/PPB.tar?dl=1'

if not os.path.exists(path_to_test_data):
    print(f"Going to fetch test data, will place it in {path_to_test_data}.")
    urllib.request.urlretrieve(test_url, ensure_path(path_to_test_data))

    with tarfile.open(f"{path_to_test_data}", "r:") as tar:
        members = [tar_mem for tar_mem in tar.getmembers()]
        tar.extractall(os.path.dirname(path_to_test_data), members=members)

else:
    print("Test data already exists.")
