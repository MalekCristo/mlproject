import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import pandas as pd
from dataclasses import dataclass
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass 

# decorator that is used to add generated special methods to classes
#decorator examines the class to find fields. A field is defined as a class variable that has a type annotation. 

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender','race_ethnicity','parental_level_of_education', 'test_preparation_course', 'lunch']

            num_pipeline = Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))

                ]
            )

            cat_pipeline = Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False)),

                ]
            )            

            logging.info('Numerical Features Scaling Completed')
            logging.info('Categorical Features Encoding Completed')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise ChildProcessError(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            logging.info('Obtaining preprocessing object')
            
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis = 1)
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis = 1)

            target_feature_train_df= train_df[target_column_name]
            target_feature_test_df= test_df[target_column_name]
            logging.info(f"Applying preprocessing object on training and testing dataframes")

            #The fit() method helps in fitting the data into a model, 
            #transform() method helps in transforming the data into a form that is more suitable for the model.
            # Fit_transform() method, on the other hand, combines the functionalities of both fit() and transform() methods in one step.

            input_feature_train_array= preprocessing_obj.fit_transform(input_feature_train_df)

            input_feature_test_array= preprocessing_obj.transform(input_feature_test_df)
            #np. c_[] concatenates arrays along second axis.
            #Similarly, np. r_[] concatenates arrays along first axis.

            train_arr = np.c_[input_feature_train_array, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_array, np.array(target_feature_test_df)]
            logging.info(f"Saving preprocessing objects")

            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path, 
                        obj = preprocessing_obj)

            return ( train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e,sys)

        
