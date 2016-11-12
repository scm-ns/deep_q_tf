import tensorflow as tf
import numpy as np



param_dict = dict(in_w , in_h , num_act , num_frm , dicount , lrn_rte , rho , rms_epsln , momntm ,
                  clp_del , freeze_interval , batch_size , network_type , update_rule,
                  batch_accum , rng , in_scl = 255.0)



class network:
    def __init__(param_dict):
        self.in_w = param_dict['in_w']
        self.in_h = param_dict['in_h']
        self.num_act = param_dict['num_act']
        self.num_frm = param_dict['num_frm']
        self.dicount = param_dict['dicont']
        self.lrn_rte = param_dict['lrn_rte'] 
        self.rho = param_dict['rho']
        self.rms_epsln = param_dict['rms_epsln'] , 
        self.mmomnt = param_dict['momntm']
                  clp_del , freeze_interval , batch_size , network_type , update_rule,
                  batch_accum , rng , in_scl = 255.0)



        # State of the network at a particular time step
        self.states = tf.placeholder(tf.float32, shape=(self.batch_size , self.num_frm , self.in_h , self.in_w))
        

        self.next_state = tf.placeholder(tf.float32 , shape=(self.batch_size , self.num_frm , self.in_h , self.in_w))

        self.q_network = tf.placeholder(
    
        self.rewards = tf.placeholder(tf.float32, shape=(self.batch_size , 1))
        
        self.actions = tf.placeholder(tf.int32 , shape=(self.batch_size , 1))
        self.terminals = tf.placeholder(tf.int32 , shape(self.batch_size , 1))





    def nips_net(self , in_w , in_h , out_d , num_fram ):
        # Feed the batch into the start of the network
        inp = tf.placeholder(tf.float32 , shape=(-1 , in_w , in_h , num_fram))
            
        w_c1 = tf.Variable(tf.truncated_normal(shape=[8,8,num_frm,16], stddev = 0.1))
        b_c1 = tf.Variable(tf.constant(0,1 , [16]))
        
        h_c1 = tf.nn.relu(tf.nn.conv2d(inp , w_c1, strides=[1,4,4,1] , padding='SAME') + b_c1)

        # size of hc1 = (in_h - 8)/4 + 1
        
        w_c2 = tf.Variable(tf.truncated_normal(shape=[4,4,16,32], stddev = 0.1))
        b_c2 = tf.Variable(tf.constant(0,1 , [32]))

        h_c2 = tf.nn.relu(tf.nn.conv2d(h_c1 , w_c2, strides=[1,2,2,1] , padding='SAME') + b_c1)

        w_fc1 = tf.Variable(tf.truncated_normal([inp_val , 256]))
        b_fc1 = tf.Variable(tf.constant(0.1 , [256])) 
    
        h_c2_flat = tf.reshape(h_c2 , [-1 , inp_val])
        h_fc1 = tf.nn.relu(tf.matmul(h_c2_flat , w_fc1) + b_fc1)
    
        w_fc2 = tf.Variable(tf.truncated_normal([256 , out_d])
        b_fc2 = tf.Variable(tf.constant(0.1 , [out_d])) 

        return tf.matmul(h_fc1 , w_fc2) + b_fc2

        

    def nature_net(self , in_w , in_h , out_d , num_fram ):
        # Feed the batch into the start of the network
        inp = tf.placeholder(tf.float32 , shape=(-1 , in_w , in_h , num_fram))
            
        w_c1 = tf.Variable(tf.truncated_normal(shape=[8,8,num_frm,32], stddev = 0.1))
        b_c1 = tf.Variable(tf.constant(0,1 , [32]))
        
        h_c1 = tf.nn.relu(tf.nn.conv2d(inp , w_c1, strides=[1,4,4,1] , padding='SAME') + b_c1)

        # size of hc1 = (in_h - 8)/4 + 1
        
        w_c2 = tf.Variable(tf.truncated_normal(shape=[4,4,32,64], stddev = 0.1))
        b_c2 = tf.Variable(tf.constant(0,1 , [64]))

        h_c2 = tf.nn.relu(tf.nn.conv2d(h_c1 , w_c2, strides=[1,2,2,1] , padding='SAME') + b_c1)
 
        w_c3 = tf.Variable(tf.truncated_normal(shape=[3,3,64,64], stddev = 0.1))
        b_c3 = tf.Variable(tf.constant(0,1 , [64]))
        h_c3 = tf.nn.relu(tf.nn.conv2d(h_c1 , w_c2, strides=[1,1,1,1] , padding='SAME') + b_c1)

        w_fc1 = tf.Variable(tf.truncated_normal([inp_val , 512]))
        b_fc1 = tf.Variable(tf.constant(0.1 , [512])) 
    
        h_c2_flat = tf.reshape(h_c2 , [-1 , inp_val])
        h_fc1 = tf.nn.relu(tf.matmul(h_c2_flat , w_fc1) + b_fc1)
    
        w_fc2 = tf.Variable(tf.truncated_normal([512 , out_d])
        b_fc2 = tf.Variable(tf.constant(0.1 , [out_d])) 

        return tf.matmul(h_fc1 , w_fc2) + b_fc2



