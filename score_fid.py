import argparse
import os
import torchvision
import torchvision.transforms as transforms
import torch
import classify_svhn
from classify_svhn import Classifier
import numpy as np
from itertools import islice
import scipy

SVHN_PATH = "svhn"
PROCESS_BATCH_SIZE = 32


def get_sample_loader(path, batch_size):
    """
    Loads data from `[path]/samples`

    - Ensure that path contains only one directory
      (This is due ot how the ImageFolder dataset loader
       works)
    - Ensure that ALL of your images are 32 x 32.
      The transform in this function will rescale it to
      32 x 32 if this is not the case.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    data = torchvision.datasets.ImageFolder(
        path,
        transform=transforms.Compose([
            transforms.Resize((32, 32), interpolation=2),
            classify_svhn.image_transform
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=2,
    )
    return data_loader


def get_test_loader(batch_size):
    """
    Downloads (if it doesn't already exist) SVHN test into
    [pwd]/svhn.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    testset = torchvision.datasets.SVHN(
        SVHN_PATH, split='test',
        download=True,
        transform=classify_svhn.image_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
    )
    return testloader


def extract_features(classifier, data_loader):
    """
    Iterator of features for each image.
    """
    with torch.no_grad():
        for x, _ in data_loader:
            h = classifier.extract_features(x).numpy()
            for i in range(h.shape[0]):
                yield h[i]

                
def calculate_fid_score(sample_feature_iterator,
                        testset_feature_iterator):
    """
    Calculating the Frechet Inception Distance to evaluate the model
    based on stored images.
    
    To avoid having a FID score that has an imaginary part, 
    we use two ways: the first is inspire by the implementation of 
    the FID score in pytorch
    (https://github.com/bioinf-jku/TTUR/blob/master/fid.py).
    This way keeps the real part if the square root if the biggest 
    imaginary component is lower than a given threshold that must be
    low enough. The second approach is to add an epsilon (small enough)
    times the identity matrix to the covariance dot product and look 
    if the FID score is real for a given small epsilon.
    
    """
    print("getting a corrected FID ...")
    
    # retrieving the list of features for all generated images
    sample_array = []
    
    for l in sample_feature_iterator:
      sample_array += [l]
      
    sample_array = np.asarray(sample_array)
    
    # retrieving the list of features extracted from the test set
    test_array = []
    
    for l in testset_feature_iterator:
      test_array += [l]
      
    test_array =  np.asarray(test_array)
    
    # estimating the mean of both distributions
    mean_sample = np.mean(sample_array, axis=0) 
    
    mean_test = np.mean(test_array, axis=0)
    
    # estimating the covariance matrix for both distribtions
    cov_sample = np.cov(sample_array, rowvar=False)
    
    cov_test = np.cov(test_array, rowvar=False)
    
    # Getting the dot product between both covariance matrices
    cov_dot_prod = np.dot(cov_sample, cov_test)
    
    
    # Using a table of values of epsilon to avoid numerical instability
    epsilon = [1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,1e-13,1e-14]
    
    print("Using 9 values of epsilon...")
    
    # creating a list for obtained FID scores that are real
    FID_scores = []
    
    for eps in epsilon:
      norm_mean = np.linalg.norm(mean_sample - mean_test)**2
      trace_cov = np.trace(cov_sample + cov_test - 2 * scipy.linalg.sqrtm(cov_dot_prod+eps))
      # getting the FID score
      FID_score = norm_mean + trace_cov
      if not isinstance(FID_score, complex):
        # appending real scores only
        FID_scores.append(round(FID_score,3))
    
    # calculating the square root of the matrix
    sqrt_cov_dot_prod = scipy.linalg.sqrtm(cov_dot_prod)
     
    # checking if sqrt_cov_dot_prod is complex
    if np.iscomplexobj(sqrt_cov_dot_prod):
      if not np.allclose(np.diagonal(sqrt_cov_dot_prod).imag, 0, atol=1e-5):
        m = np.max(np.abs(sqrt_cov_dot_prod.imag))
        raise ValueError(" An imaginary component is higher than the threshold. \
        This might mean that there is an error in the FID score calculation!")
      sqrt_cov_dot_prod = sqrt_cov_dot_prod.real
      
    norm_mean = np.linalg.norm(mean_sample - mean_test)**2
    trace_cov = np.trace(cov_sample + cov_test - 2 * sqrt_cov_dot_prod)
    # getting the FID score
    corrected_FID_score = round(norm_mean + trace_cov,3)
    
    FID_dict = {
      "corrected_FID": corrected_FID_score,
      "FID_added_eps": FID_scores
    }
    
    return FID_dict 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Score a directory of images with the FID score.')
    parser.add_argument('--model', type=str, default="svhn_classifier.pt",
                        help='Path to feature extraction model.')
    parser.add_argument('directory', type=str,
                        help='Path to image directory')
    args = parser.parse_args()

    quit = False
    if not os.path.isfile(args.model):
        print("Model file " + args.model + " does not exist.")
        quit = True
    if not os.path.isdir(args.directory):
        print("Directory " + args.directory + " does not exist.")
        quit = True
    if quit:
        exit()
    print("Test")
    classifier = torch.load(args.model, map_location='cpu')
    classifier.eval()

    sample_loader = get_sample_loader(args.directory,
                                      PROCESS_BATCH_SIZE)
    sample_f = extract_features(classifier, sample_loader)

    test_loader = get_test_loader(PROCESS_BATCH_SIZE)
    test_f = extract_features(classifier, test_loader)

    fid_score = calculate_fid_score(sample_f, test_f)
    print("FID score:", fid_score)
