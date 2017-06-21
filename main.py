'''
@author: Zezhou Sun
'''
import tensorflow as tf 
import csv 
import re
import json
import time
import math
from pprint import pprint
from gensim import corpora, models, similarities

from Classifier import *

class utils:
    """
    Some common tools to use. Like parsing sentences, write and read json file.
    """
    def parse_line(self, text_line, parse_tool="re"):
        # Use regex to find out all words
        if parse_tool == "re":
            return list(re.findall(re.compile("\w+'*\w*"), text_line.lower())) # or to use \w+-*'*\w*
    
    def write_to_json(self, obj, json_filepath="./none.json"):
        with open(json_filepath, "w") as f:
            json.dump(obj, f)
            
    def read_from_json(self, json_filepath="./none.json"):
        with open(json_filepath,"r") as f:
            temp_data = json.load(f)
        return temp_data
    
    def find_least_over_percentage(self, count_list, percentage):
        count_sum = 0 
        sub_sum = 0
        position = 0
        for i in count_list:
            count_sum += i
        for i in range(len(count_list)):
            sub_sum += count_list[i]
            if sub_sum/count_sum > percentage:
                position = i
                break
        return position
    
    def get_positions_over_percentage(self, count_list):
        result = []
        for percentage in [0.90,0.95,0.99,0.999]:
            result.append(self.find_least_over_percentage(count_list, percentage))
        return result

    ############################# RESULT #############################
    # For text: 160(90%), 216(95%), 381(99%), 746(99.9%)
    # For Summary: 8(90%), 9(95%), 12(98%), 19(99.9%)

class dictionary(utils):
    """
    Class to count words and build dictionary for reviews.
    """
    # Build a word dictionary
    word_dict = {}
    word_dict_rev = {}
    word_dict_freq = {}
    gensim_dict = None
    
    json_filepath = "./word_dict.json"
    
    def count_word(self, text_line):
        """
        count word in a given string, and add word to word_dict_freq count the occurence of words
        """
        word_list = self.parse_line(text_line)
        for word in word_list:
            if word in self.word_dict_freq:
                self.word_dict_freq[word] += 1
            else:
                self.word_dict_freq[word] = 1
    
    def generate_word_dict(self):
        """
        Sort dictionary by value, so we can get a list in ascending order
        Change: self.word_dict
                self.word_dict_rev
        """
        word_list = sorted(self.word_dict_freq, key=self.word_dict_freq.get, reverse=True)
        i = 1
        for word in word_list: 
            self.word_dict[word] = i
            self.word_dict_rev[i] = word
            i += 1
            
    def generate_gensim_dictionary(self):
        if self.word_dict != {} :
                self.gensim_dict = corpora.Dictionary([list(self.word_dict.keys())])
        else:
            raise ValueError("word_dict is null, cannot generate gensim_dictionary")
    
    def save_dictionary_to_json(self, json_filepath=json_filepath):
        """
        Save all current variables in this class to a json file
        Change: <file://json_filepath>
        """
        if json_filepath != None:
            self.json_filepath=json_filepath
        if self.word_dict != {}:
            with open(self.json_filepath, "w") as f:
                json.dump([self.word_dict, self.word_dict_rev, self.word_dict_freq], f)
        else:
            print("Trying to save an empty dictionary")
    
    def save_gensim_dictionary(self, gensim_dic_filepath="./word_dict.dict"):
        if self.gensim_dict != None:
            self.gensim_dict.save(gensim_dic_filepath)
        else:
            if self.word_dict != {} :
                self.gensim_dict = corpora.Dictionary(self.word_dict.keys())
                self.gensim_dict.save(gensim_dic_filepath)
            
    
    def load_dictionary_from_json(self, json_filepath=None):
        """
        Load all dictionary from a json file
        Change: self.word_dict
                self.word_dict_rev
                self.word_dict_freq
        """
        if json_filepath != None:
            self.json_filepath=json_filepath
        with open(self.json_filepath,"r") as f:
            temp_data = json.load(f)
        self.word_dict = temp_data[0]
        self.word_dict_rev = temp_data[1]
        self.word_dict_freq = temp_data[2]
        
    def load_gensim_dictionary(self, gensim_dic_filepath="./word_dict.dict"):
        self.gensim_dict = corpora.Dictionary.load(gensim_dic_filepath)
    
    def turn_word_to_number(self,the_list):
        """
        By using existing word_dict dictionary map a list of words to a list of numbers
        Return: list (same shape as the_list)
        """
        if len(self.word_dict) == 0:
            self.load_dictionary_from_json()
        return [self.word_dict[i] for i in the_list]

class manipulate_comment_file(dictionary):
    """
    Main class to store data and store models, and provide some query methods
    """
    comment_data = []
    train_data = []
    validate_data = []
    test_data = []
    field_name = []
    current_form = "None"
    corpus = {"summary":[],"text":[]}
    models = {"summary":{}, "text":{}}
    similarity = {"summary":None, "text":None}
    
    max_summary_length = 0
    max_text_length = 0
    text_length_count = []
    summary_length_count = []
    filepath="./example.csv"
    
    def __init__(self, filepath="./train.csv"):
        self.filepath = filepath
    def initialize(self):
        self.comment_data=[]
        self.train_data = []
        self.validate_data = []
        self.test_data = []
        self.field_name = []
        self.max_summary_length = 0
        self.max_text_length = 0
        self.text_length_count = []
        self.summary_length_count = []
    
    def read_csv_file(self):
        # Read csv review file
        self.initialize()
        with open(self.filepath, "r",encoding='utf-8') as csvfile:
            data = csv.reader(csvfile, delimiter=",")
            self.field_name = next(data)
            for row in data:
                self.comment_data.append(row)
        self.current_form = "raw file with only comment_data"        
    
    def generate_word_dict(self):
        if self.comment_data == {}:
            self.read_csv_file()
        for i in range(len(self.comment_data)):
            self.count_word(self.comment_data[i][5])
            self.count_word(self.comment_data[i][6])
        dictionary.generate_word_dict(self)
        dictionary.save_dictionary_to_json(self)

    def count_line_length(self):
        # count the maximum word length in summary and text, and store them in a list to form histogram 
        self.read_csv_file()
        for row in self.comment_data:
            sum_len = len(row[5].split())
            text_len = len(row[6].split())
            self.max_summary_length = max(self.max_summary_length, sum_len)
            self.max_text_length = max(self.max_text_length, text_len)
        text_length_count = [0 for _ in range(self.max_text_length)]
        summary_length_count = [0 for _ in range(self.max_summary_length)]
        for row in self.comment_data:
            summary_length_count[sum_len] += 1
            text_length_count[text_len] += 1
    
    def split_data(self, percentage_of_vali=0.1, with_validate=True):
        print("Going to split data with ",with_validate, "validation")
        if len(self.comment_data)==0:
            self.read_csv_file()
        validate_extract_interval = int(1/percentage_of_vali)
        validate_counter = 0
        # Clear all existing list to avoid redundance
        self.train_data=[];self.validate_data=[];self.test_data=[];
        
        for ith in range(len(self.comment_data)):
            if self.comment_data[ith][4] == '':
                # Data without level are test 
                self.test_data.append(self.comment_data[ith])
            else:
                if with_validate:
                    if validate_counter == validate_extract_interval:
                    # take one element as validation per interval
                        self.validate_data.append(self.comment_data[ith])
                        validate_counter = 0
                    else:
                        self.train_data.append(self.comment_data[ith])
                        validate_counter += 1
                else:
                    self.train_data.append(self.comment_data[ith])
        self.current_form = "Split csv raw file, with validate or without validate"
                    
    def set_all_data_aslist(self, list_of_data):
        if len(list_of_data) == 4:
            self.comment_data = list_of_data[0]
            self.train_data   = list_of_data[1]
            self.validate_data= list_of_data[2]
            self.test_data    = list_of_data[3] 
        elif len(list_of_data) == 3:
            self.comment_data = list_of_data[0]
            self.train_data   = list_of_data[1]
            self.validate_data= []
            self.test_data    = list_of_data[2]
        else:
            print("Input a list with length ", len(list_of_data))
    
    def transform_append_product_people_info_to_position(self, position=5, add_product_id=True, add_user_id=True):
        """
        Add product information into consideration. 
        This is definitely bring extra distance when evaluate two review, but could work. 
        Make sure all product id and user id and things at position are strings
        That means, data at position should be a string
        Input: 
            position: int, should be 5(add to summary) or 6(add to text)
        """
        result = [[row for row in self.comment_data], [row for row in self.train_data], [row for row in self.validate_data], [row for row in self.test_data]]
        dt = [self.comment_data, self.train_data, self.validate_data, self.test_data]
        for dt_index in range(len(dt)):
            for i in range(len(dt[dt_index])):
                if isinstance(dt[dt_index][i][position], str):
                    if add_product_id and add_user_id:
                        if isinstance(dt[dt_index][i][3], str) and isinstance(dt[dt_index][i][8], str):
                            result[dt_index][i][position] = dt[dt_index][i][3]+" "+dt[dt_index][i][8]+" "+dt[dt_index][i][position]
                        else:
                            raise ValueError("Product ID and User ID are not strings, cannot add to position")
                    elif add_product_id:
                        if isinstance(dt[dt_index][i][3], str):
                            result[dt_index][i][position] = dt[dt_index][i][3]+" "+dt[dt_index][i][position]
                        else:
                            raise ValueError("Product ID are not strings, cannot add to position")
                    elif add_user_id:
                        if isinstance(dt[dt_index][i][8], str):
                            result[dt_index][i][position] = dt[dt_index][i][8]+" "+dt[dt_index][i][position]
                        else:
                            raise ValueError("User ID are not strings, cannot add to position")
                    else:
                        print("You plan to adding nothing, why call me ?!")
                else:
                    raise ValueError("At position "+str(position)+" there should be string, but detected other thing")
        self.current_form = "Added Product ID or User ID to position "+str(position)
        return result
         
    def transform_parse_summ_and_text_using_dictionary(self):
        result = [[row for row in self.comment_data], [row for row in self.train_data], [row for row in self.validate_data], [row for row in self.test_data]]
        dt = [self.comment_data, self.train_data, self.validate_data, self.test_data]
        if self.word_dict_freq == {}:
            self.load_dictionary_from_json()
        for dt_index in range(len(dt)):
            for i in range(len(dt[dt_index])):
                if isinstance(dt[dt_index][i][5], str):
                    result[dt_index][i][5] = self.turn_word_to_number(self.parse_line(dt[dt_index][i][5]))
                if isinstance(dt[dt_index][i][6], str):
                    result[dt_index][i][6] = self.turn_word_to_number(self.parse_line(dt[dt_index][i][6]))
                if isinstance(dt[dt_index][i][4], str) and dt[dt_index][i][4] != '':
                    result[dt_index][i][4] = int(dt[dt_index][i][4])
        self.current_form = "Split data using word_dict.json turned to list of numbers"
        return result
    
    def transform_semantic_basic(self):
        result = [[row for row in self.comment_data], [row for row in self.train_data], [row for row in self.validate_data], [row for row in self.test_data]]
        dt = [self.comment_data, self.train_data, self.validate_data, self.test_data]
        words_to_remove = set("for a of the and to in on at or br so".split())
        
        if self.word_dict_freq == {}:
            self.load_dictionary_from_json()
        for dt_index in range(len(dt)):
            for i in range(len(dt[dt_index])):
                if isinstance(dt[dt_index][i][5], str):
                    result[dt_index][i][5] = [word for word in (self.parse_line(dt[dt_index][i][5])) if ((word not in words_to_remove) and (self.word_dict_freq[word]>1))]
                if isinstance(dt[dt_index][i][6], str):
                    result[dt_index][i][6] = [word for word in (self.parse_line(dt[dt_index][i][6])) if ((word not in words_to_remove) and (self.word_dict_freq[word]>1))]
                if isinstance(dt[dt_index][i][4], str) and dt[dt_index][i][4] != '':
                    result[dt_index][i][4] = int(dt[dt_index][i][4])
        self.current_form = "Split data turned to list of words and removed some low frequency words & irrelevant words"
        return result
    
    def transform_to_vectors_using_gensim_dictionary(self, with_validate=True):
        if with_validate:
            result = [[row for row in self.comment_data], [row for row in self.train_data], [row for row in self.validate_data], [row for row in self.test_data]]
            dt = [self.comment_data, self.train_data, self.validate_data, self.test_data]
        else:
            result = [[row for row in self.comment_data], [row for row in self.train_data], [row for row in self.test_data]]
            dt = [self.comment_data, self.train_data, self.test_data]
        words_to_remove = set("for a of the and to in on at or br so".split())
        if self.word_dict_freq == {}:
            self.load_dictionary_from_json()
        for dt_index in range(len(dt)):
            for i in range(len(dt[dt_index])):
                if isinstance(dt[dt_index][i][5], str):
                    result[dt_index][i][5] = self.gensim_dict.doc2bow([word for word in (self.parse_line(dt[dt_index][i][5])) if ((word not in words_to_remove) and (self.word_dict_freq[word]>1))])
                if isinstance(dt[dt_index][i][6], str):
                    result[dt_index][i][6] = self.gensim_dict.doc2bow([word for word in (self.parse_line(dt[dt_index][i][6])) if ((word not in words_to_remove) and (self.word_dict_freq[word]>1))])
                if isinstance(dt[dt_index][i][4], str) and dt[dt_index][i][4] != '':
                    result[dt_index][i][4] = int(dt[dt_index][i][4])
        self.current_form = "Split data turned to gensim word vectors"
        return result
    
    def generate_gensim_models(self, num_of_lsi_topics=5):
        if self.corpus["summary"] == [] or self.corpus["text"] == []:
            raise ValueError("Error corpus, no available corpus exist, cannot used to generate gensim models")
        tfdif_summ = models.TfidfModel(self.corpus["summary"])
        tfdif_text = models.TfidfModel(self.corpus["text"])
        print("Finished tfdif wrapper initializer for summary and text")
        lsi_after_tfdif_summ  = models.LsiModel(tfdif_summ[self.corpus["summary"]], id2word=self.gensim_dict, num_topics = num_of_lsi_topics)
        lsi_after_tfdif_text  = models.LsiModel(tfdif_text[self.corpus["text"]], id2word=self.gensim_dict, num_topics = num_of_lsi_topics)
        print("Finished lsi wrapper initializer for summary and text")
        self.models["summary"]={"tfdif":tfdif_summ, "lsi_after_tfdif":lsi_after_tfdif_summ}
        self.models["text"] = {"tfdif":tfdif_text, "lsi_after_tfdif":lsi_after_tfdif_text}
        print("Saved models in class")
        return [tfdif_summ, tfdif_text, lsi_after_tfdif_summ, lsi_after_tfdif_text]
    
    def generate_gensim_corpus(self, with_validate=True):
        """
        Require train_data already become vector after transform_to_vectors_using_gensim_dictionary
        """
        self.split_data(with_validate=with_validate)
        for comp in self.corpus:
            self.corpus[comp]=[]
        for row in self.train_data:
#             print("train data length",len(self.train_data), row)
            self.corpus["summary"].append(row[5])
            self.corpus["text"].append(row[6])
        print("Generated list of summary corpus at corpus[summary], length: ", len(self.corpus["summary"]))
        print("Generated list of    text corpus at corpus[text]   , length: ", len(self.corpus["text"]))
        self.save_serialize_corpus()
        self.load_serialize_corpus()
    
    def generate_similarity_structure(self, model_to_use, corpus_to_use="text"):
        if self.similarity[corpus_to_use] == None:
            self.similarity[corpus_to_use] = similarities.MatrixSimilarity(model_to_use[self.corpus[corpus_to_use]])
        
    def run_similarity(self, input_data, corpus_to_use="text"):
        """
        Require similarity_structure generated before use. 
        Input data:
            Input data should after analysis of model, the model used in input_data should be the same as similarity structure
        """
        if self.similarity[corpus_to_use] != None and isinstance(input_data, list):
            return self.similarity[corpus_to_use][input_data]
        else:
            raise ValueError("Null similarity_structure or error input_data type")
        
    def run_data_in_model(self, input_data, model_to_use):
        return model_to_use[input_data]
        
    def save_similarity_structure(self,corpus_to_use="text"):
        if self.similarity != None:
            self.similarity[corpus_to_use].save("./similarity_structure_"+corpus_to_use+".index")
            print("Saved similarity structure to ","similarity_structure_"+corpus_to_use)
    
    def load_similarity_structure(self,corpus_to_use="text"):
        self.similarity[corpus_to_use] = similarities.MatrixSimilarity.load("./similarity_structure_"+corpus_to_use+".index")
        print("Loaded similarity structure from ","similarity_structure_"+corpus_to_use)
    
    def save_gensim_models(self):
        if self.models["summary"] == {} and self.models["text"] == {}:
            self.generate_gensim_models()
        for text_name in self.models:
            for model_name in self.models[text_name]:
                self.models[text_name][model_name].save("./"+text_name+"_"+model_name+".model")
        print("All models saved")
            
    
    def load_gensim_models(self):
        self.models["summary"]["tfdif"] = models.TfidfModel.load("./summary_tfdif.model")
        self.models["summary"]["lsi_after_tfdif"] = models.LsiModel.load("./summary_lsi_after_tfdif.model")
        self.models["text"]["tfdif"] = models.TfidfModel.load("./text_tfdif.model")
        self.models["text"]["lsi_after_tfdif"] = models.LsiModel.load("./text_lsi_after_tfdif.model")
        print("Successful load models tfdif and lsi_after_tfdif")
    
    def save_serialize_corpus(self):
        """Still have bug, need to detect corpus type before saving."""
        if self.corpus["summary"] != [] and isinstance(self.corpus["summary"], list) and self.corpus["text"] != [] and isinstance(self.corpus["text"], list):
            corpora.MmCorpus.serialize("./train_summary.mm", self.corpus["summary"])
            corpora.MmCorpus.serialize("./train_text.mm", self.corpus["text"])
        else:
            self.generate_gensim_corpus(with_validate=True)
            corpora.MmCorpus.serialize("./train_summary.mm", self.corpus["summary"])
            corpora.MmCorpus.serialize("./train_text.mm", self.corpus["text"])
        print("Saved corpus of summary and text to local files")
        
    def load_serialize_corpus(self):
        self.corpus["summary"] = corpora.MmCorpus("./train_summary.mm")
        self.corpus["text"] = corpora.MmCorpus("./train_text.mm")
        print("Loaded summary corpus and text corpus", self.corpus["summary"], self.corpus["text"])
    
    def save_all_data_to_local(self):
        self.write_to_json(self.comment_data, "./whole_data.json")
        self.write_to_json(self.train_data, "./train_data.json")
        self.write_to_json(self.validate_data, "./validate_data.json")
        self.write_to_json(self.test_data, "./test_data.json")
        self.save_dictionary_to_json("./word_dict.json")
        self.save_gensim_dictionary("./word_dict.dict")
        self.save_serialize_corpus()
        self.save_gensim_models()
    
    def load_all_data_from_local(self):
        self.comment_data = self.read_from_json("./whole_data.json")
        self.train_data   = self.read_from_json("./train_data.json")
        self.validate_data= self.read_from_json("./validate_data.json")
        self.test_data    = self.read_from_json("./test_data.json")
        self.load_dictionary_from_json("./word_dict.json")
        self.load_gensim_dictionary("./word_dict.dict")
        self.load_serialize_corpus()
        self.load_gensim_models()
    
    def lazy_eval_forlist(self, input_list=comment_data):
        m=0
        cycle_count=0
        length_of_input = len(input_list)
        while True:
            yield input_list[m]
            m += 1
            if m > length_of_input:
                cycle_count += 1
            m = m % length_of_input

    def lazy_eval_forlist_batch(self, input_list=comment_data, batch_size=100):
        lazy_eval_forlist_one = self.lazy_eval_forlist(input_list)
        while True:
            yield [next(lazy_eval_forlist_one) for _ in range(batch_size)]
    
    def lazy_eval_forlist_batch_summ(self, input_list=train_data, batch_size=100, fixed_size=20):
        lazy_eval_batch = self.lazy_eval_forlist_batch(input_list, batch_size)
        while True:
            data = next(lazy_eval_batch)
            yield [[(data[i][5][0:fixed_size] if len(data[i][5])>fixed_size else data[i][5]+[0 for _ in range(fixed_size-len(data[i][5]))]) for i in range(batch_size)], [int(data[i][4]) for i in range(batch_size)]]
            
    def lazy_eval_forlist_batch_text(self, input_list=train_data, batch_size=100, fixed_size=380):
        lazy_eval_batch = self.lazy_eval_forlist_batch(input_list, batch_size)
        while True:
            data = next(lazy_eval_batch)
            yield [[(data[i][6][0:fixed_size] if len(data[i][6])>fixed_size else data[i][6]+[0 for _ in range(fixed_size-len(data[i][6]))]) for i in range(batch_size)], [int(data[i][4]) for i in range(batch_size)]]
    
    def group_by_position(self, position=4):
        """
        This is return a dictionary, key is the value in position and value is rows with that key
        Input: position of train_data, integer in [0,8]
        Return: dict ({id:list of rows}) 
        """
        # make sure train_data is not empty
        if len(self.train_data) == 0:
            self.split_data(0.1)
        result_dict = {}
        # Check how many identities at that position
        for row in self.train_data:
            if row[position] in result_dict:
                result_dict[row[position]].append(row)
            else:
                result_dict[row[position]] = [row]
        return result_dict
    
    def filter_at_position(self, filter_content, position=4):
        """
        This is return a dictionary, key is the value in position and value is rows with that key
        Input: 
            filter_content: rows at that position have same content will be returned 
            position: position in train_data, integer in [0,8]
        Return: dict ({id:list of rows}) 
        """
        # make sure train_data is not empty
        if len(self.train_data) == 0:
            self.split_data(0.1)
        result_dict = {filter_content:[]}
        for row in self.train_data:
            if row[position] == filter_content:
                result_dict[filter_content].append(row)
        return result_dict

class machine_learning_classifier():
    """
    Use tensorflow to run LSTM. Different from other classifier
    """
    def get_default_gpu_session(self,fraction=0.333):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = fraction
        return tf.Session(config=config)
    
    def build_lstm_regression_model(self, shape):
        # shape is dict with keys:
        # n_steps_per_batch, n_hidden_dim, n_input_dim
        with tf.Graph().as_default() as g:
            # inputs
            X = tf.placeholder(tf.int64,
                               [None, shape['n_steps_per_batch']])
            # observed outputs
            y = tf.placeholder(tf.int64, [None])
            shape['n_input_dim']=1
            
            fc_hidden_num = 20
            with tf.variable_scope('weights'):
                wi = tf.get_variable("W_curinput_input", [shape['n_input_dim'], shape['n_hidden_dim']])
                ui = tf.get_variable("W_lastoutput_input", [shape['n_hidden_dim'], shape['n_hidden_dim']])
                bi = tf.get_variable("b_input", [1, shape['n_hidden_dim']]) #it
                wc = tf.get_variable("W_curinput_mem", [shape['n_input_dim'], shape['n_hidden_dim']])
                uc = tf.get_variable("W_lastoutput_mem", [shape['n_hidden_dim'], shape['n_hidden_dim']])
                bc = tf.get_variable("W_mem", [1, shape['n_hidden_dim']]) # c~t
                wf = tf.get_variable("W_curinput_forget",[shape['n_input_dim'], shape['n_hidden_dim']])
                uf = tf.get_variable("W_lastoutput_forget",[shape['n_hidden_dim'], shape['n_hidden_dim']])
                bf = tf.get_variable("b_forget",[1, shape['n_hidden_dim']]) #ft
                wo = tf.get_variable("W_curinput_output", [shape['n_input_dim'], shape['n_hidden_dim']])
                uo = tf.get_variable("W_lastoutput_output", [shape['n_hidden_dim'], shape['n_hidden_dim']])
                vo = tf.get_variable("W_C_output", [shape['n_hidden_dim'], shape['n_hidden_dim']])
                bo = tf.get_variable("b_output", [1, shape['n_hidden_dim']]) #ot
                C0 = tf.get_variable("C0", [1,shape['n_hidden_dim']])
                h0 = tf.get_variable("h0", [1, shape['n_hidden_dim']])
                w_fn = tf.get_variable("w_fullconnected", [shape['n_hidden_dim'], fc_hidden_num])
                b_fn = tf.get_variable("b_fullconnected", [1, fc_hidden_num])
                w_fn_ah = tf.get_variable("w_fullconnected_after_hidden",[fc_hidden_num, shape['n_unique_ids']])
                b_fn_ah = tf.get_variable("b_fullconnected_after_hidden",[1, shape['n_unique_ids']])
            
            C_t_1 = C0
            h_t_1 = h0
    #         output = []
            for t in range(shape['n_steps_per_batch']):
                x_t = X[:, t]
                x_t = tf.reshape(tf.cast(x_t, tf.float32), [-1,1])
                i_t = tf.sigmoid(tf.matmul(x_t, wi)+tf.matmul(h_t_1,ui)+bi)
                C_can_t = tf.tanh(tf.matmul(x_t, wc)+tf.matmul(h_t_1,uc)+bc)
                f_t = tf.sigmoid(tf.matmul(x_t, wf)+tf.matmul(h_t_1,uf)+bf)
                c_t = tf.multiply(i_t,C_can_t) + tf.multiply(f_t, C_t_1)
                o_t = tf.sigmoid(tf.matmul(x_t, wo)+tf.matmul(c_t, vo)+tf.matmul(h_t_1, uo)+bo)
                h_t = tf.multiply(o_t , tf.tanh(c_t))
                # Prepare for the output
                C_t_1 = c_t
                h_t_1 = h_t
    #             output.append(h_t)
            
            fc_hidden = tf.add(tf.matmul(h_t, w_fn),b_fn)
            fc_afteract = tf.nn.relu(fc_hidden)
            pre_output = tf.add(tf.matmul(fc_afteract,w_fn_ah),b_fn_ah)
            output = pre_output
            #################################################################
            
            # loss and train_op
            pred = tf.argmax(output, axis = 1)+1
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y-1, logits=output)) # labels=y-1 is because y is in [1,5] but logits position is in [0,4]
            train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
            summ = tf.summary.scalar('loss_sum_%dd' % shape['n_hidden_dim'], loss)
    
        return {'inputs': [X, y], 'loss': loss, 'train_op': train_op, 'summ': summ,
                'graph': g,'pred':pred}
    
    def run_amazon_review_feature_extract(self, lazy_eval_data_getting,lazy_eval_validata_getting, feature_num, training_size, lstm_cell_num):
        # lazy_eval_data_getting should have [x,y] shape, x is input list (batch_size x n_steps_per_batch), y is expected output label(batch_size x 1)
        shape = dict(n_steps_per_batch=lstm_cell_num, n_unique_ids=5, n_hidden_dim=feature_num)
        model = self.build_lstm_regression_model(shape)
        epoch_max = 100
        batch_size = 100
        iterations_one_epoch = int(training_size/batch_size)
        with model['graph'].as_default() as g, self.get_default_gpu_session(0.9) as sess:
            sess.run(tf.global_variables_initializer())
            sum_writer = tf.summary.FileWriter("./tb", g)
            for epoch_i in range(epoch_max):
                for i in range(iterations_one_epoch):
                    # train LSTM
                    dt = next(lazy_eval_data_getting)
                    train_feed_dict = dict(zip(model['inputs'], dt)) 
                    to_compute = [model['train_op'], model['summ'], model['loss']]
                    _, summ, loss_val = sess.run(to_compute, train_feed_dict)
                    sum_writer.add_summary(summ, epoch_i*iterations_one_epoch+i)
                    if i % 100 == 0:
                        # do prediction
                        dv = next(lazy_eval_validata_getting)
                        train_feed_dict = dict(zip(model['inputs'],dv))
                        to_compute = [model['pred']]
                        pred = sess.run(model['pred'], {model['inputs'][0]:dv[0]})
    #                     print(pred)
                        accurate_num = 0
                        for j in range(len(dv[1])):
                            if dv[1][j] == pred[j]:
                                accurate_num += 1
                        accu = accurate_num/len(pred)
                        print("Epoch ",epoch_i, " Iteration ", epoch_i*iterations_one_epoch+i, ", with loss ", loss_val, "validate accuracy is ",accu, ", and prediction is ",pred)
                        sum_writer.flush()

class normal_classifiers(utils):
    """
    A class to use lots of different classifiers to get final result
    """
    classifiers = []
    classifiers_weight = []
    descending_update_rate = 0 # need to be set before training
    def set_classifiers(self, classifiers_list):
        if isinstance(classifiers_list, list):
            self.classifiers=classifiers_list
        else:
            print("Error in setting classifiers, classifiers_list should be a list")
        
    def get_result(self, input_data):
        """
        By using classifier in classifiers, get prediction result. 
        Return:
            list [rounded prediction after weight, [all predictions given by classifiers]]
        """
        if len(self.classifiers_weight) != self.classifiers:
            self.classifiers_weight = [1/len(self.classifiers) for _ in range(len(self.classifiers))]
        result = []
        for cls in self.classifiers:
            result.append(cls.run(input_data))
        weighted_result=round(sum([i*j for i, j in zip(result, self.classifiers_weight)]))
        return [round(weighted_result), result]
    
    def run_test_data(self, mcf_class, test_flag=False, des_filepath="./des.csv", get_data_id_filepath="./test.csv", round_name="default"):
        """
        test_flag only used in testing model. Should set to False when actual running. 
        """
        with open(des_filepath, "w", newline="") as csvfile, open(get_data_id_filepath,"r") as testfile:
            csvwriter = csv.writer(csvfile, delimiter=",")
            csvreader = csv.reader(testfile, delimiter=",")
            csvwriter.writerow(next(csvreader))
            i = 0
            full_len = len(mcf_class.test_data)
            for row in mcf_class.test_data:
                result = self.get_result(row)
                csvwriter.writerow([int(row[2]), int(result[0])])
#             for row in csvreader:
#                 filter_dict = mcf_class.filter_at_position(str(row[0]), position=2)
#                 for data in filter_dict:
#                     if filter_dict[data] == []:
#                         raise ValueError("Record not found")
#                     print(data, filter_dict[data])
#                     result = self.get_result(filter_dict[data])
#                 csvwriter.writerow([row[0], result[0]])
                i += 1
                if i % int(full_len*0.001) == 0:
                    print("Round ",round_name," Evaluation ",100*i/full_len,"% finished")
                if test_flag:
                    if i > 500:
                        break
                
    
    def calculate_accuracy(self, input_data_row_list, update_weight_flag=False):
        if not isinstance(input_data_row_list, list):
            print("Error input in calculate_accurary, input should be a list")
            return
        full_length = len(input_data_row_list)
        correct_num = 0
        for row in input_data_row_list:
            return_of_evalaution = self.get_result(row)
            if int(row[4]) == int(return_of_evalaution[0]):
                correct_num += 1
            if update_weight_flag:
                self.update_weight(row[4], return_of_evalaution[1])
        return correct_num/full_length
    
    def update_rate_convergence(self, start_rate=0.1, converge_speed=0.0001):
        i = 0
        while True:
            yield start_rate*math.exp(-i*converge_speed)
            i += 1
        
    def update_weight(self, correct_result, classifiers_result):
        if len(classifiers_result) != len(self.classifiers_weight):
            print("Error result input")
            return
        correct_result = int(correct_result)
        temp_weight = []
        for i in range(len(classifiers_result)):
            temp_weight.append(self.classifiers_weight[i]*1.2*math.exp(-0.9*abs(correct_result-classifiers_result[i])))
        # Normalize the weight
        sum_weight = sum(temp_weight)
        for i in range(len(self.classifiers_weight)):
            self.classifiers_weight[i] = temp_weight[i]/sum_weight

if __name__ == "__main__":
    time0 = time.time()
    
    ### BLOCK 1: Read data from csv file and build a class to store it, this is needed all the time
    data = manipulate_comment_file("./train.csv")
    data.read_csv_file() # Read csv file
    data.set_all_data_aslist(data.transform_append_product_people_info_to_position(5, add_product_id=True, add_user_id=True))
    with_validate_flag = False
    use_saved_similarity_structure = False
    data.split_data(0.1, with_validate=with_validate_flag) # split data to train, validate and test, if with_validate=False, then no validate data will generate
    
    ### BLOCK 2: If this is first time running, run this
    data.generate_word_dict() # Generate word dictionary using all summaries and texts
    data.generate_gensim_dictionary() # Generate gensim dictionary using word dictionary generated above
    data.save_gensim_dictionary() # Save gensim dictionary to local, so next time only need to load from local
    data.set_all_data_aslist(data.transform_to_vectors_using_gensim_dictionary(with_validate=with_validate_flag)) # transform all data (including training data, validate data, test data and original all data), with_validate should be the same as above
    data.generate_gensim_corpus(with_validate=with_validate_flag) # Generate stream like gensim corpus using gensim dictionary, with_validate should be the same as above
    data.generate_gensim_models(num_of_lsi_topics=372) # Generate gensim LSA model. The number of topics Get from Johnson-Lindenstrause Lemma
    data.save_gensim_models() # Save models to local files so next time can run directly
    use_saved_similarity_structure = False

    ### BLOCK 3: If all files already exist in local (including all the dictionaries), run this
#     data.load_gensim_dictionary()
#     data.load_serialize_corpus()
#     data.load_gensim_models()
#     data.set_all_data_aslist(data.transform_to_vectors_using_gensim_dictionary(with_validate=with_validate_flag)) # transform all data (including training data, validate data, test data and original all data), with_validate should be the same as above
#     use_saved_similarity_structure = True
#     print(data.corpus["summary"], data.corpus["text"])    
#     print(data.models["summary"]["lsi_after_tfdif"].print_topics(5))
#     print(data.models["text"]["lsi_after_tfdif"].print_topics(5))
    
    ### BLOCK 4: This is block is not recommended. Takes about 10 Gb memory
#     data.save_all_data_to_local()
#     data.load_all_data_from_local()

    ### BLOCK 5: For training classifier weight and try test. And use this to compare between classifiers
#     le_get_train = data.lazy_eval_forlist_batch(input_list=data.train_data, batch_size=100)
#     ct.descending_update_rate=ct.update_rate_convergence(converge_speed=0.001)
#     number_epoch = 5
#     time1 = time.time()
#     for _ in range(number_epoch):
#         accu = ct.calculate_accuracy(next(le_get_train), update_weight_flag=True)
#         print("Accuracy: ", accu)
#     print("Weight for classifiers: ",ct.classifiers_weight)
    
#     ct.run_test_data(data, test_flag=True, des_filepath="./only_text_result.csv")
#
#     le_get_train = data.lazy_eval_forlist_batch(input_list=data.validate_data, batch_size=100)
#     ct.descending_update_rate=ct.update_rate_convergence(converge_speed=0.001)
#     for _ in range(100):
#         accu = ct.calculate_accuracy(next(le_get_train), update_weight_flag=False)
#         print("Predict accuracy: ", accu)

    ### BLOCK 6: Set classifiers weight and try test, also can let it learn training weight by using validate data
    # Use highest similarity score get from text only
#     ct = normal_classifiers() # 1.08682
#     ct.set_classifiers(classifiers_list=[classifier_semantic_analysis_highest(data, use_summary=False, use_saved_similarity=use_saved_similarity_structure)]) #classifier_semantic_analysis_highest(data), classifier_semantic_analysis_average(data, 10)
#     ct.run_test_data(data, test_flag=False, des_filepath="./only_text_result.csv", round_name="only_text")
    # Use highest similarity score get from summary only
#     ct = normal_classifiers() # 1.40877
#     ct.set_classifiers(classifiers_list=[classifier_semantic_analysis_highest(data, use_summary=True, use_saved_similarity=use_saved_similarity_structure)]) #classifier_semantic_analysis_highest(data), classifier_semantic_analysis_average(data, 10)
#     ct.run_test_data(data, test_flag=False, des_filepath="./only_summary_result.csv", round_name="only_summary")
    # Use the average score of highest similarity get from summary and text
#     ct = normal_classifiers() 
#     ct.set_classifiers(classifiers_list=[classifier_semantic_analysis_highest(data, use_summary=False, use_saved_similarity=use_saved_similarity_structure),\
#                                          classifier_semantic_analysis_highest(data, use_summary=True, use_saved_similarity=use_saved_similarity_structure)]) #classifier_semantic_analysis_highest(data), classifier_semantic_analysis_average(data, 10)
#     ct.classifiers_weight = [0.5, 0.5]
#     ct.run_test_data(data, test_flag=False, des_filepath="./text_summary_highest_result.csv", round_name="text+summary")
    # Use highest similarity in text, highest similarity in summary and average of highest few similarity in summary to predict. Weight of them is 0.4, 0.3, 0.3
    ct = normal_classifiers() 
    ct.set_classifiers(classifiers_list=[classifier_semantic_analysis_highest(data, use_summary=False, use_saved_similarity=use_saved_similarity_structure),\
                                         classifier_semantic_analysis_highest(data, use_summary=True, use_saved_similarity=use_saved_similarity_structure), \
                                         classifier_semantic_analysis_average(data, 5, use_summary=True, use_saved_similarity=use_saved_similarity_structure)]) 
    ct.classifiers_weight = [0.4, 0.3, 0.3]
    ct.run_test_data(data, test_flag=False, des_filepath="./text_summary_ave_mixture.csv", round_name="text+summary+ave_summ")
    
    ### BLOCK 7: Try to use machine learning, but failed. All predictions finally are 5. Should change model. 
#    # Use text for learning
#     ledata = data.lazy_eval_forlist_batch_text(data.train_data, batch_size=100)
#     vali_ledata = data.lazy_eval_forlist_batch_text(data.validate_data, batch_size=100)
#     train_ml = machine_learning_classifier()
#     train_ml.run_amazon_review_feature_extract(ledata, vali_ledata, feature_num=10, training_size=len(data.train_data), lstm_cell_num=380)
#    # Use summary for learning
#     ledata = data.lazy_eval_forlist_batch_summ(data.train_data, batch_size=1)
#     vali_ledata = data.lazy_eval_forlist_batch_summ(data.validate_data, batch_size=1)
#     train_ml = machine_learning_classifier()
#     train_ml.run_amazon_review_feature_extract(ledata, vali_ledata, feature_num=10, training_size=len(data.train_data), lstm_cell_num=20)
    ### 
    
    ### BLOCK 8: Only for test and estimate running time. No need to use this part
#     time2 = time.time()
#     print("Cost time ",time1-time0," ,in loading, ", time2-time1, " in running, average per record is ", (time2-time1)/(100*number_epoch), " and estimate running time for test data is ",(time2-time1)*len(data.test_data)/(100*number_epoch))

