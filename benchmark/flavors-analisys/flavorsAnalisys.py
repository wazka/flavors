import os
import dataset as d
import models as m

#using only first device
os.environ["CUDA_VISIBLE_DEVICES"]="0"


datasetPath = 'C:\\Users\\alber\\Projects\\flavors-results\\P100-keys-top145\\findRandomDataset_shortConfigs.pkl'

m.sequentialDense(
    datasetPath, 
    'C:\\Users\\alber\\Projects\\flavors-results\\P100-keys-top145\\model.h5',
    'C:\\Users\\alber\\Projects\\flavors-results\\P100-keys-top145\\history.pkl',
    epochs = 2)

