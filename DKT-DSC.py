import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import time
import csv
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
from math import sqrt
import os
from numpy import array
from sklearn.cross_validation import KFold
import math
import pandas as pd


model_name = 'DKT-DSC'
data_name= 'Assist_09' # 



# flags
tf.flags.DEFINE_float("epsilon", 0.1, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("l2_lambda", 0.3, "Lambda for l2 loss.")
tf.flags.DEFINE_float("learning_rate",1e-2, "Learning rate")
tf.flags.DEFINE_float("max_grad_norm", 20.0, "Clip gradients to this norm.")
tf.flags.DEFINE_float("keep_prob", 0.6, "Keep probability for dropout")
tf.flags.DEFINE_integer("hidden_layer_num", 1, "The number of hidden layers (Integer)")
tf.flags.DEFINE_integer("hidden_size", 400, "The number of hidden nodes (Integer)")
tf.flags.DEFINE_integer("evaluation_interval", 5, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 16, "Batch size for training.")
tf.flags.DEFINE_integer("problem_len", 20, "length for each time interval")
tf.flags.DEFINE_integer("num_cluster", 7, "length for each time interval")
tf.flags.DEFINE_integer("epochs", 20, "Number of epochs to train for.")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("train_data_path", 'data/'+data_name+'_train.csv', "Path to the training dataset")
tf.flags.DEFINE_string("test_data_path", 'data/'+data_name+'_test.csv', "Path to the testing dataset")
tf.flags.DEFINE_boolean("model_name", model_name, "model used")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def add_gradient_noise(t, stddev=1e-3, name=None):

    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

class StudentModel(object):

    def __init__(self, is_training, config):
        self._batch_size = batch_size = config.batch_size
        self.num_skills = num_skills = config.num_skills        
        self.hidden_size = size = FLAGS.hidden_size
        self.num_steps = num_steps = config.num_steps
        
        current_size = (num_skills*2)
        next_size = num_skills
        sr_size = 10
        
        output_size = ((FLAGS.num_cluster+1))
        
        self.current = tf.placeholder(tf.int32, [batch_size, num_steps], name='current')  
        self.next = tf.placeholder(tf.int32, [batch_size, num_steps], name='next')
        
        self._target_id = target_id = tf.placeholder(tf.int32, [None])
        self._target_correctness = target_correctness = tf.placeholder(tf.float32, [None])
        final_hidden_size = size

        hidden_layers = []
        for i in range(FLAGS.hidden_layer_num):
            final_hidden_size = size/(i+1)
            hidden1 = tf.nn.rnn_cell.LSTMCell(final_hidden_size, state_is_tuple=True)
            if is_training and config.keep_prob < 1:
                hidden1 = tf.nn.rnn_cell.DropoutWrapper(hidden1, output_keep_prob=FLAGS.keep_prob)
            hidden_layers.append(hidden1)

        cell = tf.nn.rnn_cell.MultiRNNCell(hidden_layers, state_is_tuple=True)
        
         
        

        #one-hot encoding
        current = tf.reshape(self.current, [-1])
        with tf.device("/cpu:0"):
            labels = tf.expand_dims(current, 1)
       	    indices = tf.expand_dims(tf.range(0, batch_size*num_steps, 1), 1)
            concated = tf.concat([indices, labels],1)
            current = tf.sparse_to_dense(concated, tf.stack([batch_size*num_steps, current_size]), 1.0, 0.0)
            current.set_shape([batch_size*num_steps, current_size])
            c_data = tf.reshape(current, [batch_size, num_steps, current_size])
            slice_c_data = tf.split(c_data, num_steps, 1) 
        
        next = tf.reshape(self.next, [-1])    
        with tf.device("/cpu:0"):
            labels = tf.expand_dims(next, 1)
       	    indices = tf.expand_dims(tf.range(0, batch_size*num_steps, 1), 1)
            concated = tf.concat([indices, labels],1)
            next = tf.sparse_to_dense(concated, tf.stack([batch_size*num_steps, next_size]), 1.0, 0.0)
            next.set_shape([batch_size*num_steps, next_size])
            x_data = tf.reshape(next, [batch_size, num_steps, next_size])
            slice_x_data = tf.split(x_data, num_steps, 1)
            
            
               
        
        input_embed_l = []
        for i in range(num_steps):             
            c = tf.squeeze(slice_c_data[i], 1)
            x = tf.squeeze(slice_x_data[i], 1)
            sr = tf.squeeze(slice_sr_data[i], 1)
            
            t1= tf.concat([c,x], 1)
            input_embed_l.append(t1)
            
            
        input_embed= tf.stack(input_embed_l)        
        input_size=int(input_embed[0].get_shape()[1])
        x_input = tf.reshape(input_embed, [-1, input_size])        
        x_input = tf.split(x_input, num_steps, 0)

        
        outputs, state = rnn.static_rnn(cell, x_input, dtype=tf.float32)
        output = tf.reshape(tf.concat(outputs,1), [-1, int(final_hidden_size)])
        sigmoid_w = tf.get_variable("sigmoid_w", [final_hidden_size, output_size])
        sigmoid_b = tf.get_variable("sigmoid_b", [output_size])
        logits = tf.matmul(output, sigmoid_w) + sigmoid_b
        logits = tf.reshape(logits, [-1])        
        selected_logits = tf.gather(logits, self.target_id)
        self._all_logits = logits
        print("==> [Tensor Shape] logits\t",logits.get_shape()) 

        #make prediction
        self._pred = tf.sigmoid(selected_logits)

        # loss function
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_logits, labels=target_correctness))
        self._cost = cost = loss

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def input_data(self):
        return self._input_data

    @property
    def auc(self):
        return self._auc

    @property
    def pred(self):
        return self._pred

    @property
    def target_id(self):
        return self._target_id

    @property
    def target_correctness(self):
        return self._target_correctness

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def all_logits(self):
        return self._all_logits

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

class HyperParamsConfig(object):
    """Small config."""
    init_scale = 0.05    
    num_skills = 0
    num_steps = FLAGS.problem_len
    batch_size = FLAGS.batch_size
    max_grad_norm = FLAGS.max_grad_norm
    max_max_epoch = FLAGS.epochs
    keep_prob = FLAGS.keep_prob
        



def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def k_means_clust(session, train_students, test_students, max_stu, max_seg, num_clust, num_iter,w=50):
    identifiers=2    
    max_stu=int(max_stu)
    max_seg=int(max_seg)
    cluster= {}
    data=[]
    for ind,i in enumerate(train_students):        
        data.append(i[:-identifiers])
    
    data = array(data)    
    points = tf.constant(data)    
    
    centroids = tf.Variable(tf.random_shuffle(points)[:num_clust, :])
    points_e = tf.expand_dims(points, axis=0) 
    centroids_e = tf.expand_dims(centroids, axis=1) 
    distances = tf.reduce_sum((points_e - centroids_e) ** 2, axis=-1)
    indices = tf.argmin(distances, axis=0)
    clusters = [tf.gather(points, tf.where(tf.equal(indices, i))) for i in range(num_clust)]   
    new_centroids = tf.concat([tf.reduce_mean(clusters[i], reduction_indices=[0]) for i in range(num_clust)], axis=0)
    # update centroids
    assign = tf.assign(centroids, new_centroids)
    session.run(tf.global_variables_initializer())
    for j in range(10):
        clusters_val, centroids_val, _ = session.run([clusters, centroids, assign])


    # cluster for training data 
    for ind,i in enumerate(train_students):
        inst=i[:-identifiers]
        min_dist=float('inf')
        closest_clust=None            
        for j in range(num_clust):
            if euclideanDistance(inst,centroids_val[j],w)< min_dist:
               cur_dist=euclideanDistance(inst,centroids_val[j],w)
               if cur_dist<min_dist:
                  min_dist=cur_dist
                  closest_clust=j                  
        cluster[int(i[-2]),int(i[-1])]=closest_clust

        
    # cluster for testing data 
    for ind,i in enumerate(test_students):
        inst=i[:-identifiers]
        min_dist=float('inf')
        closest_clust=None 
        for j in range(num_clust):
            if euclideanDistance(inst,centroids_val[j],w)< min_dist:
               cur_dist=euclideanDistance(inst,centroids_val[j],w)
               if cur_dist<min_dist:
                  min_dist=cur_dist
                  closest_clust=j
        cluster[int(i[-2]),int(i[-1])]=closest_clust
    return cluster


def run_epoch(session, m, students, max_stu, cluster, run_type, eval_op, verbose=False):
    """Runs the model on the given data."""   
    index = 0
    pred_labels = []
    actual_labels = []
    all_all_logits = []
    limit = 20
    
    if (run_type=='test'):      
       limit=5  
       
       
      
    
    while(index < len(students)):
          x = np.zeros((m.batch_size, m.num_steps))
          c = np.zeros((m.batch_size, m.num_steps))
          
          
          target_ids = []
          target_correctness = [] 
          max_seg=0
        
          student = students[index]
          student_id = student[0][0]
          problem_ids = student[1]
          correctness = student[2]
          
          
          
          
          if len(problem_ids)%FLAGS.problem_len:
             max_seg= (len(problem_ids)//FLAGS.problem_len)+1
          
          if (len(problem_ids)>limit):
             for i in range(m.batch_size):
                 for j in range(m.num_steps):                     
                     current_indx= i*m.num_steps +j                  
                     target_indx = current_indx+1
                     seg_id= target_indx//FLAGS.problem_len
                                    
                     if (seg_id>0 and seg_id< max_seg):
                        cluster_id= cluster[student_id,(seg_id)]+1
                     else:
                          cluster_id= 1
                     
                     #print(str(student_id)+'_'+str(seg_id)+'_'+str(cluster_id))
                     if target_indx < len(problem_ids):
                        current_id = int(problem_ids[current_indx])
                        target_id = int(problem_ids[target_indx])
                        label_index = 0
                        correct = int(correctness[current_indx])
                        
                        
                        
                        
                        if target_id > 0:
                           if( correct == 0):
                              label_index = current_id
                              
                              
                           else:
                                label_index =(current_id + m.num_skills)
                               
                                
                                
                           c[i, j] = label_index
                           x[i, j] = target_id                           
                           burffer_space=i*m.num_steps*(FLAGS.num_cluster+1)+j*(FLAGS.num_cluster+1)
                           t_ind=burffer_space+ int(cluster_id)
                           target_ids.append(t_ind)                        
                           target_correctness.append(int(correctness[target_indx]))
                           actual_labels.append(int(correctness[target_indx]))
                        
                     
          index += 1
                    
          pred, _, all_logits = session.run([m.pred, eval_op, m.all_logits], feed_dict={
            m.current: c, m.next: x, m.target_id: target_ids, m.target_correctness: target_correctness})
            
          
          for i, p in enumerate(pred):
              pred_labels.append(p)
             
          all_all_logits.append(all_logits)
        
     
    rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
    fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    r2 = r2_score(actual_labels, pred_labels)

    return rmse, auc, r2, np.concatenate(all_all_logits)


def read_data_from_csv_file(path):
    config = HyperParamsConfig()
    problem_len = FLAGS.problem_len
    
    f_data = open(path, 'r')
    max_seg = 0
    max_skills = 0
    max_attempts= 0
    students = []
    students_all = []
    studentids = []
    for lineid, line in enumerate(f_data):
        if lineid % 3 == 0:
           stu = line.split(',')
           stu_id=int(stu[1])
           studentids.append(stu_id)
        elif lineid % 3 == 1:
             q_tag_list = line.split(',')
        elif lineid % 3 == 2:
             answer_list = line.split(',')
             
             s1=[stu_id]
             t_all=(s1,q_tag_list,answer_list)
             students_all.append(t_all)
             
             tmp_max_attempts = int(len(q_tag_list))
             if(tmp_max_attempts> max_attempts):
                max_attempts = tmp_max_attempts
             
             if len(q_tag_list) > problem_len:
                n_split = len(q_tag_list) // problem_len
                
                if len(q_tag_list) % problem_len:
                   n_split += 1
                   tmp_max_seg = int(n_split)
                   if(tmp_max_seg > max_seg):
                      max_seg = tmp_max_seg
                else:
                     n_split = 1
                     
                for k in range(n_split):
                    q_container = []
                    a_container = []
                    if k == n_split - 1:
                       end_index = len(answer_list)
                    else:
                         end_index = (k+1)*problem_len
                    
                    for i in range(k*problem_len, end_index):
                        q_container.append(int(q_tag_list[i])+1)
                        a_container.append(int(answer_list[i]))
                        if(len(q_container)>problem_len):
                           q_container= [] 
                           a_container=[]
                           
                    if len(q_container)>0:
                       s1=[stu_id,k]
                       tuple_data=(s1,q_container,a_container)
                       students.append(tuple_data)
                       tmp_max_skills = max(q_container)
                       if(tmp_max_skills > max_skills):
                          max_skills = tmp_max_skills
                    
    f_data.close()   
    
     
    max_steps= np.round(max_attempts)
    
    max_stu=max(studentids)+1
    index=0
    cluster_data = []
    xtotal = np.zeros((max_stu,max_skills+1))
    x1 = np.zeros((max_stu,max_skills+1))
    x0 = np.zeros((max_stu,max_skills+1))
    while(index < len(students)):
         student = students[index]
         stu_id = int(student[0][0])
         seg_id = int(student[0][1])
         problem_ids = student[1]
         correctness = student[2]
         for j in range(len(problem_ids)):
             key =problem_ids[j]
             xtotal[stu_id,key] +=1
             if(int(correctness[j]) == 1):
                x1[stu_id,key] +=1
             else:
                  x0[stu_id,key] +=1
         xsr=[x/y for x, y in zip(x1[stu_id], xtotal[stu_id])]
         xfr=[x/y for x, y in zip(x0[stu_id], xtotal[stu_id])]
         
         x=np.nan_to_num(xsr)-np.nan_to_num(xfr)
         x=np.append(x, stu_id)
         x=np.append(x, seg_id)
         cluster_data.append(x)         
         index+=1
         
    del students 
    
    return students_all, cluster_data, studentids, max_skills, max_seg, max_steps 


def main(unused_args):
    config = HyperParamsConfig()
    
    train_students, train_cluster_data, train_ids, train_max_skills,train_max_seg, train_max_steps = read_data_from_csv_file(FLAGS.train_data_path)
    test_students, test_cluster_data, test_ids, test_max_skills, test_max_seg, test_max_steps = read_data_from_csv_file(FLAGS.test_data_path)    
    max_skills=max([int(train_max_skills),int(test_max_skills)])+1    
    config.num_skills = max_skills    
    max_stu= max(train_ids+test_ids)+1
    max_seg=max([int(train_max_seg),int(test_max_seg)])+1
    #config.num_steps= max([int(train_max_steps),int(test_max_steps)]) 
    config.num_steps = 100
    print('Shape of train data : %s,  test data : %s, max_step : %s ' % (len(train_students), len(test_students) , config.num_steps) )  
    with tf.Graph().as_default():
         session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
         global_step = tf.Variable(0, name="global_step", trainable=False)
         starter_learning_rate = FLAGS.learning_rate
         learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 3000, 0.96, staircase=True)
         optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=FLAGS.epsilon)
         
         with tf.Session(config=session_conf) as session:              
              cluster =k_means_clust(session, train_cluster_data, test_cluster_data, max_stu, max_seg, FLAGS.num_cluster, max_skills)
              initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
              
              # training model
              with tf.variable_scope("model", reuse=None, initializer=initializer):
                   m = StudentModel(is_training=True, config=config)
              # testing model
              with tf.variable_scope("model", reuse=True, initializer=initializer):
                   mtest = StudentModel(is_training=False, config=config)
              grads_and_vars = optimizer.compute_gradients(m.cost)
              grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v)
              for g, v in grads_and_vars if g is not None]
              grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
              train_op = optimizer.apply_gradients(grads_and_vars, name="train_op", global_step=global_step)
              session.run(tf.initialize_all_variables())
              j=1
              for i in range(config.max_max_epoch):
                  rmse, auc, r2, _ = run_epoch(session, m, train_students, max_stu, cluster, 'train', train_op, verbose=False)
                  print("Epoch: %d Train Metrics:\n rmse: %.3f \t auc: %.3f \t r2: %.3f \n" % (i + 1, rmse, auc, r2))
                  if((i+1) % FLAGS.evaluation_interval == 0):
                     rmse, auc, r2, all_logits = run_epoch(session, mtest, test_students, max_stu, cluster, 'test', tf.no_op(), verbose=True)
                     print("Epoch: %d Test Metrics:\n rmse: %.3f \t auc: %.3f \t r2: %.3f \n" % (j, rmse, auc, r2))
                     j+=1
                        
       
             
               
if __name__ == "__main__":
    tf.app.run()
