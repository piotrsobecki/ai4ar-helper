# AI4AR Helper Package

## Installation


Python

```
python3 setup.py install
```


PIP

```
python -m pip install -e git+https://github.com/piotrsobecki/ai4ar-helper.git#egg=ai4ar
```



## Usage

The package provides interfaces to enable more readable navigation and processing of AI4AR dataset.

Examples below, see the [main file (helper.py)](src/ai4ar/helper.py) for implementation.



```
from ai4ar import Dataset, Case

# Create Dataset API object
dataset = Dataset('data/raw')


# Navigate single case (patient data)
case = dataset['001']

# Print all images available in the case
print(case.images_keys())

# Visualize all images starting with path
case.visualize('lesion_labels/lesion1/t2w/')

# Combine the masks (ie. when more labellers were involved) - in this example, require agreement of two raters
combined_t2w_mask = case.image('lesion_labels/lesion1/t2w', combine=True, cache=False, combine_pp=required_agreement(2))

# Clinical metadata (contents of the .csv file)
dataset.clinical_metadata

# Radiological metadata (contents of the .csv file), extended with paths to available lesion labels
dataset.radiological_metadata 


# The same works for the single case 
case.radiological_metadata()
case.clinical_metadata()

```


See the [tests notebook](tests/tests.ipynb) for example usage.