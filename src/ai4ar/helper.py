from benedict import benedict
import SimpleITK as sitk
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import logging as logging

log = logging.getLogger(__name__)


modalities = ['adc', 'cor', 'hbv', 'sag', 't2w', 'dce1', 'dce2', 'dce3', 'dce4', 'dce5', 'dce6']
anatomicals =  ['afs', 'cz', 'pg', 'pz', 'sv_l', 'sv_r', 'tz']

    
# Select the slice with the largest mask
def select_slice(data):
    max_slice = 0
    max_size = 0
    for i, slice in enumerate(data):
        size = slice.sum()
        if size > max_size:
            max_size = size
            max_slice = i
    return max_slice

def _visualize(case, pattern:str=None):
    images = case.images_match(pattern)
    if len(images) == 0:
        print('No images found')
        return
    
    fig, axes = plt.subplots(len(images),1, figsize=(10*len(images), 25))
    if (len(images) == 1):
        axes = [axes]
        
    for image_i in range(len(images)):
        image = images[image_i]
        image_arr = image.arr()
        image_slide = select_slice(image_arr)
        axes[image_i].imshow(image_arr[image_slide], cmap='gray')
        axes[image_i].set_title('{}[{}]'.format(image.path,image_slide))
        axes[image_i].axis('off')
    plt.tight_layout()  
    plt.show()

    
class Image: 

    def __init__(self, path, file: str = None, image = None, base_image = None):
        if file is None and image is None:
            raise ValueError("Either file or image must be provided")
        self.file = file
        self.path = path
        self.image= None
        if image is not None:
            # Check if image is a numpy array
            if isinstance(image, np.ndarray):
                # Convert the image to a SimpleITK image if it is a numpy array
                self.image = sitk.GetImageFromArray(image)
                if base_image is not None:
                    self.image.CopyInformation(base_image)
            # Check if image is a SimpleITK image
            elif isinstance(image, sitk.Image): 
                self.image = image
            else:
                raise ValueError("Image must be a numpy array or SimpleITK image")

    def __str__(self):
        return "Image(path={}, file={}, image={})".format(self.path, self.file, self.image)
    
    def __repr__(self):
        return self.__str__()
    
    
            
    def sitk(self):
        if self.image is None:
            self.image = sitk.ReadImage(self.file)
        return self.image

    def arr(self):
        return sitk.GetArrayFromImage(self.sitk())
    
    def write(self, file):
        # Make the directory if it doesn't exist for the floc file
        os.makedirs(os.path.dirname(file), exist_ok=True)
        
        # Write the image to the file location or raise an error if the file exists
        if os.path.exists(file):
            raise ValueError("File {} already exists".format(file))
        
        sitk.WriteImage(self.sitk(), file)
        
    
    def clear_image_cache(self):
        self.image = None

def _read_case(case_id, data_dir):
    
    def extract_id(string):
        # Split the string on the underscores
        parts = string.split('_')
        # The id is the last part of the string, so get the last element in the list
        id = parts[-1]
        # Split the id on the period to remove the file extension
        id = id.split('.')[0]
        return id

    anatomical_labels = {}
    data = {}
    lesion_labels = {}
    
    for label in anatomicals:
        fname = "{}_{}_t2w.nii.gz".format(str(int(case_id)), label)
        floc =  os.path.join(data_dir, 'AI4AR_cont', 'Anatomical_Labels', case_id, fname)
        if os.path.exists(floc):
            anatomical_labels[label] = Image(path='anatomical_labels/'+label,file=floc)
    
    for label in modalities:
        fname = "{}_{}.mha".format(str(int(case_id)), label)
        floc = os.path.join(data_dir, 'AI4AR_cont', 'Data', case_id, fname)
        if os.path.exists(floc):
            data[label] = Image(path='data/'+label, file=floc)
    
    base_dir = os.path.join(data_dir, 'AI4AR_cont', 'Lesion_labels', case_id)
    
    for lesion_dir in os.listdir(base_dir):
        lesion_labels[lesion_dir] = {}
        
        for data_type in modalities:
            
            lesion_data_modality = os.path.join(base_dir, lesion_dir, data_type)
            
            # Check if lesion_data_modality exists
            if os.path.exists(lesion_data_modality):
                lesion_labels[lesion_dir][data_type] = {}
                
                for lesion_annotation in os.listdir(lesion_data_modality):
                    lesion_annotation_id = extract_id(lesion_annotation)
                    floc = os.path.join(lesion_data_modality, lesion_annotation)
                    path = 'lesion_labels/'+lesion_dir+'/'+data_type+'/'+lesion_annotation_id
                    lesion_labels[lesion_dir][data_type][lesion_annotation_id]  = Image(path=path, file=floc)
    return {
        'anatomical_labels':anatomical_labels, 
        'data':data, 
        'lesion_labels': lesion_labels
    }
    
    
# Function that combines the masks from multiple raters, and returns a mask with the minimum number of raters that must have marked the lesion as present
def required_agreement(n):
    def pp(arr):
        arr_out = arr>=n
        #Convert to int 
        arr_out = arr_out.astype(int)
        return arr_out
    return pp

class Case:
    
    def __init__(self, dataset, case_id: str):
        self.dataset = dataset
        self.case_id = case_id
        self.k_sep='/'
        self.data = benedict(_read_case(case_id, self.dataset.data_dir), keypath_separator=self.k_sep)
        
        self.clinical_metadata_idx = self.dataset.clinical_metadata['patient_id'] == int(self.case_id)
        self.radiological_metadata_idx = self.dataset.radiological_metadata['patient_id'] == int(self.case_id)
        
    def __repr__(self):
        return "Case object with id {}".format(self.case_id)

    def __str__(self):
        return "Case object with id {}".format(self.case_id)
    
    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)
    
    def __contains__(self, key):
        return key in self.data

    def __setitem__(self, key, value):
        raise NotImplementedError("Cannot set values in a Case object")
    
    def __delitem__(self, key):
        raise NotImplementedError("Cannot delete values in a Case object")
    
    def __eq__(self, other):
        return self.data == other.data
    
    def __ne__(self, other):
        return self.data != other.data
    
    def __iter__(self):
        return self.images_keys().__iter__()   
    
    def images_keys(self):
        return [p for p in self.data.keypaths() if type(self[p]) == Image]
    
    def images_match(self, pattern):
        return [ img for img in self.data.match(pattern) if type(img) == Image ] 
    
    def clinical_metadata(self):
        return self.dataset.clinical_metadata[self.clinical_metadata_idx]
    
    def radiological_metadata(self):
        return self.dataset.radiological_metadata[self.radiological_metadata_idx]
    
    
    
    def image(self, key:str, combine=False, combine_pp = required_agreement(1), cache=True) -> Image:
        '''
        Get an image from the case. If the image is not already loaded, it will be loaded from the file.
        If combine is True, the image will be combined from all the files in the key path.
        combine_pp is a function that will be applied to the combined image array.
        
        Parameters
        ----------
        key : str
            The key path to the image
        combine : bool, optional
            If True, the image will be combined from all the files in the key path. The default is False.
        combine_pp : function, optional
            A function that will be applied to the combined image array. The default is lambda x: pp_at_least(x, 1).
        
        Returns
        -------
        Image
            The image object.
        '''            
                
        def _combine(pattern, path):
            images = self.images_match(pattern)
            if len(images)>0:
                base_image = images[0]
                out_val = np.zeros(base_image.arr().shape)
                for im in images:
                    out_val += im.arr()
                out_val = combine_pp(out_val)
                out_val = Image(path= path, image=out_val, base_image=base_image.sitk())
                return out_val
            
            return None
        
        out_val = None
        
        if key not in self.data and not combine:
            log.debug('there is no data and do not combine')
            return None
        
        if key in self.data:  
            log.debug('there is data')
            out_val = self.data[key]
            if type(out_val) is Image:
                log.debug('the data is an image')
                return out_val
            else:
                out_val = None
        
        if combine:
            log.debug('Try combine')
            path = self.k_sep.join(['combined',key])
            log.debug('path: {}'.format(path))
            if path in self.data:
                log.debug('path in data')
                out_val = self.data[path]
            else:
                cache_floc = os.path.join(self.dataset.tmp_dir, key,  'combined.nii.gz')
                out_val = None
                if cache and os.path.exists(cache_floc):
                    log.debug('from cache')
                    out_val = Image(path, cache_floc)
                else:
                    log.debug('try combine')
                    out_val = _combine(self.k_sep.join([key, '*']), path)
                    if out_val is not None and cache:
                        # Save the combined image using the sitk image library to the tmp dir within the same directory structure as the key path
                        out_val.write(cache_floc)
                if out_val is not None:
                    log.debug('set data')
                    self.data[path] = out_val
            
        #if out_val is None:
        #    raise ValueError("No image found for key {}".format(key))
        
        return out_val
            
    def visualize(self, base_path:str):
        _visualize(self, base_path)


class Dataset:
    def __init__(self, data_dir: str, tmp_dir: str = None):
        self.data_dir = data_dir
        # If tmp dir is none then create a tmp dir next to the data dir with the same name and _tmp
        if tmp_dir is None:
            tmp_dir = os.path.join(os.path.dirname(data_dir), os.path.basename(data_dir)+'_tmp')
            # Make the tmp dir if it does not exist
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)          
        
        self.tmp_dir = tmp_dir
        self.case_ids = os.listdir(os.path.join(self.data_dir, 'AI4AR_cont', 'Data'))
        self.cases = {}
        self.clinical_metadata = self._load_clinical_metadata()
        # This is the original radiological metadata
        self.radiological_metadata = self._load_radiological_metadata()
        # This is the extended radiological metadata with the lesion labels cols (images)
        self.radiological_metadata = self._load_radiological_metadata_ext()
        
        
        
    
    
    def _load_clinical_metadata(self):
        return pd.read_csv(os.path.join(self.data_dir, 'AI4A4_PCa_clinical.csv'))
    
    def _load_radiological_metadata(self):
        return  pd.read_csv(os.path.join(self.data_dir, 'AI4AR_PCa_radiological.csv'))
    
    def _load_radiological_metadata_ext(self):
        r_metadata =  self.radiological_metadata
        
        # Extension of the radiological metadata
        r_metadata_ext_floc =  os.path.join(self.tmp_dir, 'AI4AR_PCa_radiological-ext.csv')
        
        idx_cols = ['patient_id', 'lesion_id', 'radiologist_id']
        
        if os.path.exists(r_metadata_ext_floc):
            r_metadata_ext = pd.read_csv(r_metadata_ext_floc)
            r_metadata = r_metadata.merge(r_metadata_ext, on=idx_cols, how='left')
        
        else :
            modality_label_cols = []
            # Init paths to lesion labels
            for modality in modalities:
                modality_col = 'label_'+modality
                modality_label_cols.append(modality_col)
                r_metadata[modality_col] = 'lesion_labels/lesion'+r_metadata['lesion_id'].astype(str)+'/'+modality+'/'+r_metadata['radiologist_id'].str.split('_').str[1]

            # Check if the image exists
            for idx, row in r_metadata.iterrows():
                case = self[str(row['patient_id']).zfill(3)]
                for col in modality_label_cols:
                    if row[col]  not in case: 
                        r_metadata.loc[idx, col] = None
            
            # Save only idx and modality label cols to the extension file
            idx_cols.extend(modality_label_cols)
            
            r_metadata[idx_cols].to_csv(r_metadata_ext_floc, index=False, header=True)
        
        return r_metadata
    
    
    def __getitem__(self, case_id: str):
        if case_id not in self.cases:
            self.cases[case_id] = Case(self, case_id)
        return self.cases[case_id]
    
    def __len__(self):
        return len(self.case_ids)
    
    def __iter__(self):
        for case_id in self.case_ids:
            yield self.__getitem__(case_id)
        
    def __repr__(self):
        return "Dataset object with {} cases".format(len(self.case_ids))
    
    def __str__(self):
        return "Dataset object with {} cases".format(len(self.case_ids))
    
    def __call__(self, case_id: str):
        return self.__getitem__(case_id)
    
    def __contains__(self, case_id: str):
        return case_id in self.case_ids
    
    def load_cases(self):
        for case_id in self.case_ids:
            self.__getitem__(case_id)
