
from benedict import benedict
import SimpleITK as sitk
import os 
import numpy as np
import matplotlib.pyplot as plt 

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
    anatomical_labels_options = ['afs', 'cz', 'pg', 'pz', 'sv_l', 'sv_r', 'tz']
    data = {}
    data_options = ['adc', 'cor', 'hbv', 'sag', 't2w', 'dce1', 'dce2', 'dce3', 'dce4', 'dce5', 'dce6']
    lesion_labels = {}
    lesion_options = data_options
    
    
    for label in anatomical_labels_options:
        fname = "{}_{}_t2w.nii.gz".format(str(int(case_id)), label)
        floc =  os.path.join(data_dir, 'AI4AR_cont', 'Anatomical_Labels', case_id, fname)
        if os.path.exists(floc):
            anatomical_labels[label] = {'file': floc}
    
    for label in data_options:
        fname = "{}_{}.mha".format(str(int(case_id)), label)
        floc = os.path.join(data_dir, 'AI4AR_cont', 'Data', case_id, fname)
        if os.path.exists(floc):
            data[label] = {'file': floc}
    
    base_dir = os.path.join(data_dir, 'AI4AR_cont', 'Lesion_labels', case_id)
    
    for lesion_dir in os.listdir(base_dir):
        lesion_labels[lesion_dir] = {}
        
        for data_type in lesion_options:
            
            lesion_data_modality = os.path.join(base_dir, lesion_dir, data_type)
            
            # Check if lesion_data_modality exists
            if os.path.exists(lesion_data_modality):
                lesion_labels[lesion_dir][data_type] = {}
                
                for lesion_annotation in os.listdir(lesion_data_modality):
                    lesion_annotation_id = extract_id(lesion_annotation)
                    lesion_labels[lesion_dir][data_type][lesion_annotation_id] = {
                        'file':os.path.join(lesion_data_modality, lesion_annotation)
                    }
    return {
        'anatomical_labels':anatomical_labels, 
        'data':data, 
        'lesion_labels': lesion_labels
    }


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

def _visualize(case):
        

    image_groups = {}
    for image_key in case:
        image_group = image_key.split(case.k_sep)[0]
        if image_group not in image_groups:
            image_groups[image_group] = []
        image_groups[image_group].append(image_key)
        


    # Visualize the images in the case - anatomical labels, data, and lesion labels (if present) separately using subplots for each group and image key as the title

    for i, (image_group, image_keys) in enumerate(image_groups.items()):
        fig, axes = plt.subplots(len(image_keys),1, figsize=(10*len(image_keys), 25))
        for image_key_i in range(len(image_keys)):
            image_key = image_keys[image_key_i]
            image = case.image(image_key)
            image_slide = select_slice(image)
            axes[image_key_i].imshow(image[image_slide], cmap='gray')
            axes[image_key_i].set_title('{}[{}]'.format(image_key,image_slide))
            axes[image_key_i].axis('off')
        plt.tight_layout()  
        plt.show()

def _read_image(file):
    return sitk.GetArrayFromImage(sitk.ReadImage(file))
    
class Case:
    
    def __init__(self, case_id: str, data_dir: str):
        self.case_id = case_id
        self.data_dir = data_dir
        self.k_sep='/'
        self.data = benedict(_read_case(case_id, data_dir), keypath_separator=self.k_sep)
        
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
        return set([ self.k_sep.join(path.split(self.k_sep)[:-1]) for path in self.data.keypaths() if path.endswith('file') or path.endswith('image') ])
    
    def images_initialized_keys(self):
        return set([ self.k_sep.join(path.split(self.k_sep)[:-1]) for path in self.data.keypaths() if path.endswith('image') ])
    
    def image(self, key, combine=False):
                
        def _combine(key):
            files = self.data[key].match('*.file')
            images = [_read_image(file) for file in files]
            if len(images)>0:
                out_val = np.zeros(images[0].shape)
                for im in images:
                    out_val += im
                return out_val
            return None
        
        data = self.data[key] 
        if 'image' in data:
            return data['image']
        
        out_val = None
        
        if 'file' in data:
            out_val = _read_image(data['file'])
            self.data[key]['image'] = out_val
        elif combine:
            if '_combined' in data:
                out_val = data['_combined']['image']
            else:
                out_val = _combine(key)
                if out_val is not None:
                    self.data[key]['_combined'] = {'image':out_val}
            
        if out_val is None:
            raise ValueError("No image found for key {}".format(key))
        
        return out_val

    def summarize(self):
        for x in sorted(self.images_keys()):
            print('Image: {}'.format(x))
            
    def visualize(self):
        _visualize(self)


class Dataset:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.case_ids = os.listdir(os.path.join(self.data_dir, 'AI4AR_cont', 'Data'))
        self.cases = {}
        
    def __getitem__(self, case_id: str):
        if case_id not in self.cases:
            self.cases[case_id] = Case(case_id, self.data_dir)
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
