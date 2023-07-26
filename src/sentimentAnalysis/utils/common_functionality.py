import os
import re
import requests
import emoji
import string
import pickle
from nltk.corpus import stopwords
from box.exceptions import BoxValueError
import yaml
from src.sentimentAnalysis import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import spacy
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer


exclude=string.punctuation
# try:
#     tokenizer = spacy.load("en_core_web_sm")
# except: # If not present, we download
#     spacy.cli.download("en_core_web_sm")
#     tokenizer = spacy.load("en_core_web_sm")
nltk.download('stopwords')
stop_word=stopwords.words("english")

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"






def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logger.info("Exception occured while saving pickle file:"+str(e))


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logger.info('Exception Occured in load_object function utils:'+str(e))




def text_preprocessing(text):

    #Lowercasing
    text = text.lower()

    # Removing HTML tags
    pattern=re.compile('<.*?>')
    text = pattern.sub(r'',text)


    #Removing URL
    pattern=re.compile('http://\S+|https://\S+')
    text = pattern.sub(r'',text)

    #Removing punctuations
    text = text.translate(str.maketrans('','',exclude))

    
    #Removing stop words
    new_word=[]
    for word in text.split():
        if word.lower() in stop_word:
            new_word.append('')
        else:
            new_word.append(word)
    text = " ".join(new_word)


    return text




def sent_to_vector(word2vec_modl,sent):
    vector_size = word2vec_modl.wv.vector_size
    wv_res = np.zeros(vector_size)
    ctr = 1
    for w in sent:
        if w in word2vec_modl.wv:
            ctr +=1
            wv_res +=word2vec_modl.wv[w]
    wv_res = wv_res/ctr
    return wv_res
        


