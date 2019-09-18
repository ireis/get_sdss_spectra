import csv
from os import system
from astropy.io import fits
from joblib import Parallel, delayed

#import sfdmap
import numpy
import astropy
import os
from tqdm import tqdm_notebook as tqdm

#global m
#m = sfdmap.SFDMap('/storage/home/itamarreis/quasars/i_dmat/dustmap/')
global PATH_0
PATH_0 = '../data/'




def objecsts_ids_list(csv_path):
    """
    :param csv_path: path to csv file
    :return: list of objects in the csv file
    """
    print('loading object id csv file')
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        objects_ids = list(reader)

    print('done loading csv file')

    return objects_ids[1:]

def spectrum_file_name(plate,mjd,fiber):
    """
    #Returns the spectrum file name for a single object
    #Example link: https://data.sdss.org/sas/dr13/sdss/spectro/redux/26/spectra/lite/1678/spec-1678-53433-0034.fits
    """

    fname = "spec-"+plate.zfill(4)+'-'+mjd+'-'+fiber.zfill(4)+".fits"

    return fname

def spectrum_path(plate, run2d, lite = True):
    """
    returns the link to the spectrum fits file from SDSS website
    https://data.sdss.org/sas/dr13/sdss/spectro/redux/26/spectra/lite/1678/spec-1678-53433-0034.fits
    https://data.sdss.org/sas/dr14/sdss/spectro/redux/26/spectra/lite/1678/spec-1678-53433-0034.fits
    """

    DR = 'dr15/'

    if lite:
        lite = 'lite/'
    else:
        lite = ''

    if run2d == 'v5_9_0':
        path  = DR + "sdss/spectro/redux/v5_9_0/spectra/" + lite
    elif run2d == '26':
        path = DR + "sdss/spectro/redux/26/spectra/" + lite
    elif run2d == '103':
        path = DR + "sdss/spectro/redux/103/spectra/" + lite
    elif run2d == '104':
        path = DR + "sdss/spectro/redux/104/spectra/" + lite
    elif run2d == 'v5_10_0':
        path = DR + "sdss/spectro/redux/v5_10_0/spectra/" + lite
    else:
        print('\n')
        print('Error:')
        print('Bad run2d:', run2d)
        print('\n')
    path = path + plate.zfill(4) + "/"

    return path


def spectrum_link(plate, run2d, spec_fname, lite = True):
    """
    returns the link to the spectrum fits file from SDSS website
    https://data.sdss.org/sas/dr13/sdss/spectro/redux/26/spectra/lite/1678/spec-1678-53433-0034.fits
    """

    DR = 'dr15/'

    if lite:
        lite = 'lite/'
    else:
        lite = ''

    if run2d == 'v5_9_0':
        link  = "https://data.sdss.org/sas/" + DR +"eboss/spectro/redux/v5_9_0/spectra/" + lite + plate.zfill(4) + "/" + spec_fname
    elif run2d == '26':
        link = "https://data.sdss.org/sas/" + DR +"sdss/spectro/redux/26/spectra/" + lite + plate.zfill(4) + "/" + spec_fname
    elif run2d == '103':
        link = "https://data.sdss.org/sas/" + DR +"sdss/spectro/redux/103/spectra/" + lite + plate.zfill(4) + "/" + spec_fname
    elif run2d == '104':
        link = "https://data.sdss.org/sas/" + DR +"sdss/spectro/redux/104/spectra/" + lite + plate.zfill(4) + "/" + spec_fname
    elif run2d == 'v5_10_0':
        link = "https://data.sdss.org/sas/" + DR +"eboss/spectro/redux/v5_10_0/spectra/" + lite + plate.zfill(4) + "/" + spec_fname
    else:
        print('\n')
        print('Error:')
        print('Bad run2d:', run2d)
        print('\n')

    return link


def folder_verify(path_1):


    if  not os.path.isdir(PATH_0 + path_1):
                path_1_split = path_1.split("/")
                path_1_tmp = '/'.join(path_1_split)
                finish_ind  = len(path_1_split)
                start_ind = finish_ind
                while not os.path.isdir(PATH_0 + path_1_tmp):
                    start_ind = start_ind - 1
                    path_1_tmp = '/'.join(path_1_split[:start_ind])

                start_ind = start_ind + 1
                for ind in range(start_ind, finish_ind):
                    path_1_tmp = '/'.join(path_1_split[:ind])
                    command = 'mkdir ' + PATH_0 + path_1_tmp
                    print(command)
                    os.system(command)
    return


def rm_and_download_spectra(spectra_path, spec_fname, plate,run2d, lite):


    system('rm -rf ' + PATH_0 + spectra_path + spec_fname)

    link = spectrum_link(plate,run2d, spec_fname, lite)

    #print(link)
    _ = folder_verify(spectra_path)

    command = "wget " + link + ' -P ' + PATH_0 + spectra_path
    #print(command)
    system(command)
    return

def download_single_spectrum(plate,mjd,fiber,run2d, lite = True):

    spectra_path = spectrum_path(plate,run2d, lite)
    spec_fname = spectrum_file_name(plate,mjd,fiber)

    # If file already exists do nothing
    try:
        hdulist = fits.open(PATH_0 + spectra_path + spec_fname)
        hdulist.close()
        print('Valid file exists', PATH_0 + spectra_path + spec_fname)
    except (OSError, astropy.utils.exceptions.AstropyUserWarning):
        print('Downloading new file', PATH_0 + spectra_path + spec_fname)
        rm_and_download_spectra(spectra_path, spec_fname, plate,run2d,lite)

    return PATH_0 + spectra_path + spec_fname

def get_run2d(df):

    plates_103 = list(numpy.load('../get_data/plates_103.npy'))
    plates_104 = list(numpy.load('../get_data/plates_104.npy'))
    plates_26 = list(numpy.load('../get_data/plates_26.npy'))
    plates_v5_10_0 = list(numpy.load('../get_data/plates_v5_10_0.npy'))
    run2d = []

    for p in df['plate'].values.astype(int):
        if p in plates_v5_10_0:
            run2d += ['v5_10_0']
        elif p in plates_103:
            run2d += ['103']
        elif p in plates_104:
            run2d += ['104']
        elif p in plates_26:
            run2d += ['26']
        else:
            print(run2d, p)


    df['run2d'] = run2d


    return df

def download_spectra(df):
    plate_arr  = df['plate'].values
    mjd_arr  = df['mjd'].values
    fiberid_arr  = df['fiberid'].values
    df = get_run2d(df)
    run2d_arr  = df['run2d'].values
    nof_objects = df.shape[0]

    #_ =  Parallel(n_jobs=-1, verbose = 10)(delayed(download_single_spectrum)(str(plate), str(mjd), str(fiber), 'v5_10_0')  for plate,mjd,fiber in zip(plate_arr,mjd_arr,fiberid_arr))
    for plate,mjd,fiber,run2d in zip(plate_arr,mjd_arr,fiberid_arr,run2d_arr):
        download_single_spectrum(str(plate), str(mjd), str(fiber), run2d)

    return
