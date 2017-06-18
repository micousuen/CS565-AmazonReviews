'''
@author: Zezhou Sun
'''
import random
import gensim
import heapq

class classifier:
    """
    Interface class for a classifier provided to normal_classifiers
    """
    Usage = ""
    def __init__(self, mcf_class):
        Usage = "Do something here to init this classifier"
    def run(self, input_data): 
        return 0

class classifier_statistic(classifier): # 40-50% accuracy
    """
    Heuristic, Use the distribution of score to randomly generate scores
    """
    train_dict = {}
    total_records = 0
    records_distribution = {}
    def __init__(self, mcf_class):
        self.train_dict = mcf_class.group_by_position(4)
        for thekey in self.train_dict:
            self.total_records += len(self.train_dict[thekey])
            self.records_distribution[thekey] = len(self.train_dict[thekey])
    
    def run(self, input_data):
        random_value = random.random()*self.total_records
        for thekey in self.records_distribution:
            random_value -= self.records_distribution[thekey]
            if random_value < 0:
                return int(thekey)

class classifier_product_average(classifier):
    """
    Heuristic, Use the average of same product review history to predict new review
    """
    product_dict = {}
    def __init__(self, mcf_class):
        self.product_dict = mcf_class.group_by_position(3)
        
    def run(self, input_data):
        if input_data[3] in self.product_dict:
            product_reviews = self.product_dict[input_data[3]]
        else:
            product_reviews = []
        product_score_sum = 0
        product_reviews_num = 0
        for review in product_reviews:
            helpful_weight = (2*int(review[1])-int(review[0]))+1
            product_score_sum += helpful_weight*int(review[4])
            product_reviews_num += helpful_weight
        if product_reviews_num != 0:
            fixed_pred = round(product_score_sum/product_reviews_num) # Get history average score
        else:
            fixed_pred = 5 # Because 5 star is the highest frequency score
        return fixed_pred
    
class classifier_semantic_analysis_highest:
    """
    Use generated LSA model to run similarity query. 
    Use score in review which have highest similarity with current one as return
    """
    local_mcf_ref = None
    is_summary = True
    def __init__(self, mcf_class, use_summary=True):
        self.Usage = "Use gensim in mcf_class to get new similar document and score"
        self.local_mcf_ref = mcf_class
        self.is_summary = use_summary
        if self.is_summary:
            self.local_mcf_ref.generate_similarity_structure(model_to_use=self.local_mcf_ref.models["summary"]["lsi_after_tfdif"], corpus_to_use="summary")
        else:
            self.local_mcf_ref.generate_similarity_structure(model_to_use=self.local_mcf_ref.models["text"]["lsi_after_tfdif"], corpus_to_use="text")
            
    def run(self, input_data):
        if self.is_summary:
            data_after_model = self.local_mcf_ref.run_data_in_model(input_data[5], model_to_use=self.local_mcf_ref.models["summary"]["lsi_after_tfdif"])
            similarity_list  = self.local_mcf_ref.run_similarity(data_after_model, "summary")
        else:
            data_after_model = self.local_mcf_ref.run_data_in_model(input_data[6], model_to_use=self.local_mcf_ref.models["text"]["lsi_after_tfdif"])
            similarity_list  = self.local_mcf_ref.run_similarity(data_after_model, "text")
        max_index = similarity_list.argmax()
        highest_similarity_score = int(self.local_mcf_ref.train_data[max_index][4])
        return highest_similarity_score
    
class classifier_semantic_analysis_average:
    """
    Use generated LSA model to run similarity query. 
    Use the average of scores in reviews which are first n reviews with highest similarity
    """
    local_mcf_ref = None
    num_of_highest_score_to_average = 10
    is_summary = True
    def __init__(self, mcf_class, num_of_higest_score_to_average, use_summary = True):
        self.Usage = "Use gensim in mcf_class to get new similar document and score"
        self.local_mcf_ref = mcf_class
        self.is_summary = use_summary
        if self.is_summary:
            self.local_mcf_ref.generate_similarity_structure(model_to_use=self.local_mcf_ref.models["summary"]["lsi_after_tfdif"], corpus_to_use="text")
        else:
            self.local_mcf_ref.generate_similarity_structure(model_to_use=self.local_mcf_ref.models["text"]["lsi_after_tfdif"], corpus_to_use="text")
        self.num_of_highest_score_to_average = num_of_higest_score_to_average

    def run(self, input_data):
        if self.is_summary:
            data_after_model = self.local_mcf_ref.run_data_in_model(input_data[5], model_to_use=self.local_mcf_ref.models["summary"]["lsi_after_tfdif"])
            similarity_list  = self.local_mcf_ref.run_similarity(data_after_model, "summary")
        else:
            data_after_model = self.local_mcf_ref.run_data_in_model(input_data[6], model_to_use=self.local_mcf_ref.models["text"]["lsi_after_tfdif"])
            similarity_list  = self.local_mcf_ref.run_similarity(data_after_model, "text")
        first_N_largest = heapq.nlargest(self.num_of_highest_score_to_average, range(len(similarity_list)), similarity_list.take)
        sum_of_similarity_score = 0
        for i in first_N_largest:
            sum_of_similarity_score += int(self.local_mcf_ref.train_data[i][4])
        average_similarity_score = round(sum_of_similarity_score/self.num_of_highest_score_to_average)
        return int(average_similarity_score)
    
    
