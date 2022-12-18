
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


def visualize_case(case_data, path=[]):
    
    def select_slice(data):
        # Select the slice with the largest mask
        max_slice = 0
        max_size = 0
        for i, slice in enumerate(data):
            size = slice.sum()
            if size > max_size:
                max_size = size
                max_slice = i
        return data[max_slice]
    
    fig = None
    for i, (key, value) in enumerate(case_data.items()):
        if isinstance(value, dict):
            # Recursively process the sub-dictionary
            visualize_case(value, path + [key])
        elif value is not None:
            if fig is None:
                
                fig_width = 5 * len(case_data)
                # Create a figure with subplots
                fig, ax = plt.subplots(1,len(case_data), figsize=(fig_width, 5))

            # Select a slice for the current value
            slice = select_slice(value)
            
            ax_p = ax
            if len(case_data)>1:
                ax_p = ax[i]
            # Plot the data for the current value
            ax_p.imshow(slice, cmap='gray')
            # Set the plot title using the path to the value in the case_data dictionary
            ax_p.set_title('/'.join(path + [key]))
    if fig is not None:
        plt.show()



    

def read_case(data_dir, case_id):
    
    def read_image(file):
        return sitk.GetArrayFromImage(sitk.ReadImage(file))
        
    def extract_id(string):
        # Split the string on the underscores
        parts = string.split('_')
        # The id is the last part of the string, so get the last element in the list
        id = parts[-1]
        # Split the id on the period to remove the file extension
        id = id.split('.')[0]
        return id


    import os 
    # Read the .nii image with SimpleITK:
    anatomical_labels = {
        
    }
    data = {
        
    }
    
    lesion_labels = {
        
    }
    
    anatomical_labels_options = ['afs', 'cz', 'pg', 'pz', 'sv_l', 'sv_r', 'tz']
    
    data_options = ['adc', 'cor', 'hbv', 'sag', 't2w', 'dce1', 'dce2', 'dce3', 'dce4', 'dce5', 'dce6']
    
    lesion_dir_options = data_options
    
    for label in anatomical_labels_options:
        fname = "{}_{}_t2w.nii.gz".format(str(int(case_id)), label)
        anatomical_labels[label] = read_image(os.path.join(data_dir, 'AI4AR_cont', 'Anatomical_Labels', case_id, fname))
    
    for label in data_options:
        fname = "{}_{}.mha".format(str(int(case_id)), label)
        data[label] = read_image(os.path.join(data_dir, 'AI4AR_cont', 'Data', case_id, fname))
    
    
    base_dir = os.path.join(data_dir, 'AI4AR_cont', 'Lesion_labels', case_id)
    
    for lesion_dir in os.listdir(base_dir):
        lesion_labels[lesion_dir] = {}
        
        for data_type in lesion_dir_options:
            
            lesion_data_modality = os.path.join(base_dir, lesion_dir, data_type)
            
            # Check if lesion_data_modality exists
            if os.path.exists(lesion_data_modality):
                lesion_labels[lesion_dir][data_type] = {}
                
                lesion_annotations = os.listdir(lesion_data_modality)
                
                for lesion_annotation in lesion_annotations:
                    
                    lesion_annotation_id = extract_id(lesion_annotation)
                    
                    file = os.path.join(lesion_data_modality, lesion_annotation)
                    
                    lesion_labels[lesion_dir][data_type][lesion_annotation_id] = read_image(file)

        
        
    return {
        'anatomical_labels':anatomical_labels, 
        'data':data, 
        'lesion_labels': lesion_labels
    }
        
        




# Print the case dict but without the data, instead in place of value print the shape of the data if the value is not a dict
def describe(case, path=[]):
    for key, value in case.items():
        if isinstance(value, dict):
            # Recursively process the sub-dictionary
            describe(value, path + [key])
        elif value is not None:
            # Print the shape of the data
            print('/'.join(path + [key]), value.shape)
        else:
            # Print None
            print('/'.join(path + [key]), value)
            