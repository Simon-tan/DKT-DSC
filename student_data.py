# The code is rewritten based on source code from tensorflow tutorial for Recurrent Neural Network.
# https://www.tensorflow.org/versions/0.6.0/tutorials/recurrent/index.html
# You can get source code for the tutorial from
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/ptb_word_lm.py
#
# There is dropout on each hidden layer to prevent the model from overfitting
#
# Here is an useful practical guide for training dropout networks
# https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
# You can find the practical guide on Appendix A

import csv
import os



def read_data_from_csv_file(file_path,file_name,file_type,student_id=1):
    
    rows = []
    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)

    new_file_path = "./new/{}_{}.csv".format(file_name,file_type)    
    output_file = open(new_file_path, 'w')
    csv_writer = csv.writer(output_file, delimiter=',')       
    
    index = 0       
    while(index < len(rows)-1):
          stu_iden=[]
          problems = int(rows[index][0])
          stu_iden.append(problems)
          stu_iden.append(student_id)          
          problem_ids = rows[index+1]          
          correctness = rows[index+2]
          
          csv_writer.writerow(stu_iden)
          csv_writer.writerow(problem_ids)
          csv_writer.writerow(correctness)
                 
          student_id+=1
          index += 3
          
    max_num_student= student_id
    print ("Finish reading and writing data")   
    return max_num_student
    


def main():    
    files = ['0910_a', '0910_b','0910_c','2015_builder','CAT']
    for file_name in files:
        train_file_path='data/'+file_name+'_train.csv'
        student_id=1 
        max_num_stud =read_data_from_csv_file(train_file_path,file_name,'train',student_id)
        test_file_path='data/'+file_name+'_test.csv'
        max_num_stud =read_data_from_csv_file(test_file_path,file_name,'test',max_num_stud)
    
    
    
               

main()
