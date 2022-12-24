
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

def _visualize(case, base_path:str=None):
    image_groups = {}
    for image_key in case:
        image_group = image_key.split(case.k_sep)[0]
        if image_group not in image_groups:
            image_groups[image_group] = []
        image_groups[image_group].append(image_key)
    
    # Visualize the images in the case - anatomical labels, data, and lesion labels (if present) separately using subplots for each group and image key as the title
    for i, (image_group, image_keys) in enumerate(image_groups.items()):
         # Filter the image keys to only include those that start with the base path
        image_keys_filtr_i = [i for i, image_key in enumerate(image_keys) if base_path is None or image_key.startswith(base_path)]
        if len(image_keys_filtr_i) == 0:
            continue
        fig, axes = plt.subplots(len(image_keys_filtr_i),1, figsize=(10*len(image_keys_filtr_i), 25))
        if (len(image_keys_filtr_i) == 1):
            axes = [axes]
        for image_key_f_i in range(len(image_keys_filtr_i)):
            image_key_i = image_keys_filtr_i[image_key_f_i]
            image_key = image_keys[image_key_i]
            image = case.image(image_key)
            image_arr = image.arr()
            image_slide = select_slice(image_arr)
            axes[image_key_f_i].imshow(image_arr[image_slide], cmap='gray')
            axes[image_key_f_i].set_title('{}[{}]'.format(image_key,image_slide))
            axes[image_key_f_i].axis('off')
        plt.tight_layout()  
        plt.show()
    
def _read_image(file):
    sitk_image = sitk.ReadImage(file)
    return sitk.GetArrayFromImage(sitk_image)
    
    
class Image: 

    def __init__(self, file: str = None, image = None, base_image = None):
        if file is None and image is None:
            raise ValueError("Either file or image must be provided")
        self.file = file
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

# Function that combines the masks from multiple raters, and returns a mask with the minimum number of raters that must have marked the lesion as present
def required_agreement(n):
    def pp(arr):
        arr_out = arr>=n
        #Convert to int 
        arr_out = arr_out.astype(int)
        return arr_out
    return pp

class Case:
    
    def __init__(self, case_id: str, data_dir: str, tmp_dir:str):
        self.case_id = case_id
        self.data_dir = data_dir
        self.tmp_dir = tmp_dir
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
    
    
    def image(self, key, combine=False, combine_pp = required_agreement(1)):
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
                
        def _combine(key):
            files = self.data[key].match('*.file')
            images = [Image(file) for file in files]
            if len(images)>0:
                base_image = images[0]
                out_val = np.zeros(base_image.arr().shape)
                for im in images:
                    out_val += im.arr()
                out_val = combine_pp(out_val)
                out_val = Image(image=out_val, base_image=base_image.sitk())
                return out_val
            
            return None
        
        data = self.data[key] 
        if 'image' in data:
            return data['image']
        
        out_val = None
        
        if 'file' in data:
            out_val = Image(data['file'])
            self.data[key]['image'] = out_val
        elif combine:
            if '_combined' in data:
                out_val = data['_combined']['image']
            else:
                floc = os.path.join(self.tmp_dir, key,  'combined.nii.gz')
                
                out_val = None
                if os.path.exists(floc):
                    out_val = Image(floc)
                else:
                    out_val = _combine(key)
                    if out_val is not None:
                        
                        # Save the combined image using the sitk image library to the tmp dir within the same directory structure as the key path
                        out_val.write(floc)

                self.data[key]['_combined'] = {'image':out_val, 'file':floc}
            
        if out_val is None:
            raise ValueError("No image found for key {}".format(key))
        
        return out_val

    def summarize(self):
        for x in sorted(self.images_keys()):
            print('Image: {}'.format(x))
            
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
        
    def __getitem__(self, case_id: str):
        if case_id not in self.cases:
            self.cases[case_id] = Case(case_id, self.data_dir, self.tmp_dir)
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
