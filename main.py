'''
@author: Zezhou Sun
'''
import tensorflow as tf 
import csv 
import re
import json
import matplotlib.pyplot as plt
import time

class utils:
    def parse_word(self, text_line):
        # Use regex to find out all words
        return list(re.findall(re.compile("\w+"), text_line.lower())) # or to use \w+-*'*\w*
    
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
    # Build a word dictionary
    word_dict = {}
    word_dict_rev = {}
    word_dict_freq = {}
    
    json_filepath = "./word_dict.json"
    
    def count_word(self, text_line):
        """
        count word in a given string, and add word to word_dict_freq count the occurence of words
        """
        word_list = self.parse_word(text_line)
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
    
    def save_dictionary_to_json(self, json_filepath=json_filepath):
        """
        Save all current variables in this class to a json file
        Change: <file://json_filepath>
        """
        if json_filepath != None:
            self.json_filepath=json_filepath
        with open(self.json_filepath, "w") as f:
            json.dump([self.word_dict, self.word_dict_rev, self.word_dict_freq], f)
    
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
    
    def turn_word_to_number(self,the_list):
        """
        By using existing word_dict dictionary map a list of words to a list of numbers
        Return: list (same shape as the_list)
        """
        if len(self.word_dict) == 0:
            self.load_dictionary_from_json()
        return [self.word_dict[i] for i in the_list]

class manipulate_comment_file(dictionary):
    comment_data = []
    train_data = []
    validate_data = []
    test_data = []
    field_name = []
    
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
    def parse_summ_and_text_using_dictionary(self):
        if self.word_dict != {}:
            for dt in [self.comment_data, self.train_data, self.validate_data, self.test_data]:
                for i in range(len(dt)):
                    if isinstance(dt[i][5], str):
                        dt[i][5] = self.turn_word_to_number(self.parse_word(dt[i][5]))
                    if isinstance(dt[i][6], str):
                        dt[i][6] = self.turn_word_to_number(self.parse_word(dt[i][6]))
                    if isinstance(dt[i][4], str) and dt[i][4] != '':
                        dt[i][4] = int(dt[i][4])
    
    def save_all_data_to_local_json(self):
        self.write_to_json(self.comment_data, "./whole_data.json")
        self.write_to_json(self.train_data, "./train_data.json")
        self.write_to_json(self.validate_data, "./validate_data.json")
        self.write_to_json(self.test_data, "./test_data.json")
    
    def load_all_data_from_local_json(self):
        self.comment_data = self.read_from_json("./whole_data.json")
        self.train_data   = self.read_from_json("./train_data.json")
        self.validate_data= self.read_from_json("./validate_data.json")
        self.test_data    = self.read_from_json("./test_data.json")
        self.load_dictionary_from_json("./word_dict.json")

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
    
    def filter_at_position(self, filter, position=4):
        """
        This is return a dictionary, key is the value in position and value is rows with that key
        Input: 
            filter: rows at that position have same content will be returned 
            position: position in train_data, integer in [0,8]
        Return: dict ({id:list of rows}) 
        """
        # make sure train_data is not empty
        if len(self.train_data) == 0:
            self.split_data(0.1)
        result_dict = {filter:[]}
        for row in self.train_data:
            if row[position] == filter:
                result_dict[filter].append(row)
        return result_dict

class machine_learning_classifier():
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

class normal_classifiers():
    a=1

if __name__ == "__main__":
    ### Read data from csv file and build a class to store it
    data = manipulate_comment_file("./train.csv")
    
    ### All these are used to generate dictionary
#     data.read_csv_file()
#     data.generate_word_dict()
    
    ### All these are used for raw csv comment file
#     data.read_csv_file() # Read csv file
#     data.split_data(0.1) # split data to train, validate and test
#     data.read_word_dict_json()
#     data.parse_summ_and_text_using_dictionary()
#     data.save_all_data_to_local_json()

    ### Use saved json file to load data
    data.load_all_data_from_local_json()

#     groupby_score = data.group_by_position(3)
#     for i in groupby_score:
#         print("The key: ",i, " length: ",len(groupby_score[i]))
#     print("The total ids in this is ", len(groupby_score))
    for fil in range(1,6,1):
        filtered = data.filter_at_position(fil, 4)
        for i in filtered:
            print("The key: ",i, " length: ",len(filtered[i]))
    
    ### Try to use machine learning, but failed. All predictions finally are 5
#     ledata = data.lazy_eval_forlist_batch_text(data.train_data, batch_size=100)
#     vali_ledata = data.lazy_eval_forlist_batch_text(data.validate_data, batch_size=100)
#     ledata = data.lazy_eval_forlist_batch_summ(data.train_data, batch_size=1)
#     vali_ledata = data.lazy_eval_forlist_batch_summ(data.validate_data, batch_size=1)
#     train_ml = machine_learning_classifier()
#     train_ml.run_amazon_review_feature_extract(ledata, vali_ledata, feature_num=10, training_size=len(data.train_data), lstm_cell_num=20)

    ### 
    
    

