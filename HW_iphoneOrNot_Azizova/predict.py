
# library
import argparse
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing import image

def prepare_image(path):
    
    loaded_image = image.load_img(path, target_size=(224, 224))
    processed_image = image.img_to_array(loaded_image)
    processed_image = np.expand_dims(processed_image, axis=0)
    processed_image /= 255.
    
    return processed_image

def run_predicting(model_path, input_folder, output_csv_name):

    print('Predictor initialization...')
    print()

    model = tf.lite.Interpreter(model_path=model_path)
    model.allocate_tensors()
    input_tensors = model.get_input_details()
    output_tensors = model.get_output_details()

    probas = []
    images_names = []

    print('Start predicting...')
    print()
    
    for subdir, dirs, files in os.walk(input_folder):
        for i, file in enumerate(files):
            input_data = prepare_image(os.path.join(subdir, file))
            model.set_tensor(input_tensors[0]['index'], input_data)
            model.invoke()
            probability = model.get_tensor(output_tensors[0]['index'])
            probas.append(probability[0][0])
            images_names.append(file)
            
            print("{} of {} images are classified".format(i+1,len(files)))
            
    print("Finalization..")
    print()
    output = pd.DataFrame(data = {'file_names': images_names, 'probability': probas})
    output.to_csv(output_csv_name, index=False)
    print()
    print("Yass! Predictions are ready. Find the results in {} file".format(output_csv_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Probability to be an iphone')
    parser.add_argument('--model', type=str, help='Path to model')
    parser.add_argument('--input', type=str, help='Path to folder with pictures')
    parser.add_argument('--output', type=str, help='Path to csv file with prediction results')
    args = parser.parse_args()
    print("model= {0} input_data= {1} output_data= {2}".format(args.model, args.input, args.output))

    # launch predicting probas
    run_predicting(args.model, args.input, args.output)

