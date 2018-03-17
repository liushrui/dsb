import warnings
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
from skimage.io import imread
from skimage.transform import resize
from skimage.morphology import closing, opening, disk
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

from rle import prob_to_rles, rle_encoding

def clean_img(x):
    return opening(closing(x, disk(1)), disk(3))


def inference(model, test_images, test_image_shapes, Id_index=-3, test_glob_path='../data/test_stage1/*/*/*'):
    # build df
    test_df = pd.DataFrame(glob(test_glob_path), columns=['path'])
    test_df['ImageId'] = test_df['path'].map(lambda x:x.split('/')[Id_index])
    test_df.sort_values(by='ImageId', inplace=True)
    test_df['shape'] = test_image_shapes
    
    test_df.index = range(0, test_df.shape[0], 1)

    predictions = model.predict(test_images / 255)
    test_df['mask'] = predictions.squeeze().tolist()
    test_df['mask'] = test_df['mask'].map(lambda x: np.array(x))

    # resize
    count = 0
    for _ in range(test_df.shape[0]):
        if test_df.loc[_, 'shape'][:2] == (256, 256):
            pass
        else:
            count += 1
            test_df.loc[_, 'mask'] = resize(np.array(test_df.loc[_, 'mask']), test_df.loc[_, 'shape'][:2], preserve_range=True)

    print(count, 'images have been resized!')
    
    #rle
    test_df['rles'] = test_df['mask'].map(clean_img).map(lambda x: list(prob_to_rles(x)))
    
    out_pred_list = []
    for _, c_row in test_df.iterrows():
        for c_rle in c_row['rles']:
            out_pred_list+=[dict(ImageId=c_row['ImageId'], 
                                 EncodedPixels = ' '.join(np.array(c_rle).astype(str)))]
    
    out_pred_df = pd.DataFrame(out_pred_list)
    print(out_pred_df.shape[0], 'regions found for', test_df.shape[0], 'images')
    
    out_pred_df_path = '../submit/' + datetime.now().strftime('%m%d_%H%M%S_') + 'predictions.csv'
    out_pred_df[['ImageId', 'EncodedPixels']].to_csv(out_pred_df_path, index = False)
    
    return test_df