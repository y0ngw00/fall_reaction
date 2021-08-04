import warnings
warnings.filterwarnings(action='ignore') 
from network import Actor
from network import Critic
from network import Discriminator
from monitor import Monitor
import argparse
import random
import numpy as np
import tensorflow as tf
import tensorboard
import pickle
import datetime
import os
import time
import sys

from IPython import embed
from copy import deepcopy
from utils import RunningMeanStd
from utils import ReplayBuffer
import utils_mpi

from tensorflow.python import pywrap_tensorflow
import scipy.integrate as integrate
import types

#define 
import tracemalloc

np.set_printoptions(threshold=sys.maxsize)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
if type(tf.contrib) != types.ModuleType:
	tf.contrib._warning = None
class AMP(object):
	def __init__(self, learning_rate_actor=1e-5, learning_rate_critic=0.001, learning_rate_disc=1e-6,learning_rate_decay=0.9993,
		gamma=0.95, gamma_sparse=0.99, lambd=0.95, grad_penalty=10,task_reward_w=0.5, epsilon=0.2):
		random.seed(int(time.time()))
		np.random.seed(int(time.time()))
		tf.set_random_seed(int(time.time()))
		
		self.learning_rate_critic = learning_rate_critic
		self.learning_rate_actor = learning_rate_actor
		self.learning_rate_disc = learning_rate_disc
		self.learning_rate_decay = learning_rate_decay
		self.epsilon = epsilon
		self.gamma = gamma
		self.gamma_sparse = gamma_sparse
		self.lambd = lambd
		self.grad_penalty = grad_penalty
		self.reward_max = 0
		self.task_reward_w=task_reward_w

	def initRun(self, pretrain, num_state, num_action,num_feature=0, num_poses=0, num_slaves=1):
		self.pretrain = pretrain

		self.num_slaves = num_slaves
		self.num_action = num_action
		self.num_state = num_state
		self.num_feature = num_feature
		self.num_poses= num_poses

		config = tf.ConfigProto()
		config.intra_op_parallelism_threads = self.num_slaves
		config.inter_op_parallelism_threads = self.num_slaves
		self.sess = tf.Session(config=config)
		
		#build network and optimizer
		name = pretrain.split("/")[-1]

		self.buildOptimize(name)
		save_list = [v for v in tf.trainable_variables() if v.name.find(name)!=-1]
		self.saver = tf.train.Saver(var_list=save_list, max_to_keep=1)
		self.step = 0

		if self.pretrain is not "":
			path = self.pretrain +"/network-0"
			self.load(path)
			li = path.split("network")
			suffix = li[-1]

			self.RMS = RunningMeanStd(shape=(self.num_state))
			self.RMS.load(li[0]+"network"+li[1]+'rms'+suffix)
			self.RMS.setNumStates(self.num_state)

	def initTrain(self, name, env, pretrain="", directory=None, 
		batch_size=256, batch_size_disc=16, steps_per_iteration=2048, num_buffer=100000,optim_frequency=1):

		self.name = name
		self.directory = directory
		self.steps_per_iteration = steps_per_iteration
		self.optim_frequency = [optim_frequency, optim_frequency * 2]

		self.batch_size = batch_size
		self.batch_size_disc = batch_size_disc
		self.pretrain = pretrain
		self.env = env
		self.num_slaves = self.env.num_slaves
		self.num_action = self.env.num_action
		self.num_state = self.env.num_state
		self.num_feature = self.env.num_feature
		self.num_poses= self.env.num_poses

		self.num_procs = utils_mpi.GetNumProcs()
		self.num_buffer = int(np.ceil(num_buffer/self.num_procs))
		self.expert_buffer = ReplayBuffer(num_buffer)
		self.agent_buffer = ReplayBuffer(num_buffer)

		self.last_target_update = 0

		config = tf.ConfigProto()
		config.intra_op_parallelism_threads = self.num_slaves
		config.inter_op_parallelism_threads = self.num_slaves
		self.sess = tf.Session(config=config)

		#Tensorboard Initialise
		TB_DIR = directory+"/logs/"
		if not os.path.exists(TB_DIR):
			os.mkdir(TB_DIR)

		self.merged_summary = tf.summary.merge_all()

		TB_DIR += datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		self.writer = tf.summary.FileWriter(TB_DIR, graph=tf.get_default_graph())
		self.writer.add_graph(self.sess.graph)


		#build network and optimizer
		self.buildOptimize(name)
		# load pretrained network
		if self.pretrain is not "":
			path = "./output/"+self.pretrain +"/network-0"
			self.load(path)
			li = path.split("network")
			suffix = li[-1]

			if len(li) == 2:
				self.env.RMS.load(li[0]+'rms'+suffix)
				self.env.RMS_disc.load(li[0]+'disc-rms-0')
			else:
				self.env.RMS.load(li[0]+"network"+li[1]+'rms'+suffix)
				self.env.RMS_disc.load(li[0]+"network"+li[1]+'disc-rms-0')
			self.env.RMS.setNumStates(self.num_state)
			self.env.RMS_disc.setNumStates(self.num_feature)

		self.printInitialSetting()
		

	def printInitialSetting(self):
		
		print_list = []
		print_list.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		print_list.append("test_name : {}".format(self.name))
		print_list.append("Total reference frame : {}".format(self.num_poses))
		print_list.append("num_slaves : {}".format(self.num_slaves))
		print_list.append("num state : {}".format(self.num_state))
		print_list.append("num action : {}".format(self.num_action))
		print_list.append("learning_rate(actor) : {}".format(self.learning_rate_actor))
		print_list.append("learning_rate(discriminator) : {}".format(self.learning_rate_disc))
		print_list.append("gamma : {}".format(self.gamma))
		print_list.append("lambd : {}".format(self.lambd))
		print_list.append("gradient_penalty : {}".format(self.grad_penalty))
		print_list.append("batch_size for Policy: {}".format(self.batch_size))
		print_list.append("batch_size for Discriminator: {}".format(self.batch_size_disc))
		print_list.append("steps_per_iteration : {}".format(self.steps_per_iteration))
		print_list.append("clip ratio : {}".format(self.epsilon))
		print_list.append("pretrain : {}".format(self.pretrain))

		for s in print_list:
			print(s)

		if self.directory is not None:
			out = open(self.directory+"parameters", "w")
			for s in print_list:
				out.write(s + "\n")
			out.close()

			out = open(self.directory+"results", "w")
			out.close()

	def createDiscriminator(self, name, grad_penalty, pose_real, pose_sim, clip=True):
		self.disc_real = Discriminator(self.sess, name, pose_real)
		self.disc_fake = Discriminator(self.sess, name, pose_sim, reuse=True)

		with tf.variable_scope(name+'_Optimize'):
			self.fake_loss =  tf.reduce_mean(tf.square(self.disc_fake.pred + 1),axis=-1)
			self.real_loss = tf.reduce_mean(tf.square(self.disc_real.pred - 1),axis=-1)
			disc_loss = 0.5* (self.fake_loss + self.real_loss)

			disc_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_disc)

			self.mean_real = tf.reduce_mean(self.disc_real.pred)
			self.mean_fake = tf.reduce_mean(self.disc_fake.pred)
			# tf.summary.scalar("Agent logits", self.mean_fake)
			# tf.summary.scalar("Expert logits", self.mean_real)

			grad_real = tf.gradients(ys=self.disc_real.pred, xs=pose_real)
			grad_reals = tf.concat(grad_real, axis=-1)
			norm_grad = tf.reduce_sum(tf.square(grad_reals), axis=-1)
			self.penalty_loss = grad_penalty * 0.5 * tf.reduce_mean(norm_grad)        

			# _grads, _ = zip(*disc_trainer.compute_gradients(self.real_loss));
			# _ , _grad_norm_real = tf.clip_by_global_norm(_grads, 0.5)
			# self.penalty_loss = grad_penalty/2 *_grad_norm_real
		loss = disc_loss + self.penalty_loss

		# tf.summary.scalar("Total Loss", loss)
		# tf.summary.scalar("Discriminator loss", disc_loss)
		# tf.summary.scalar("Gradient penalty loss", self.penalty_loss)

		# disc_vars = tf.trainable_variables()

		disc_vars = tf.trainable_variables(scope = name+'_Discriminator')
		self.disc_grads = tf.gradients(loss, disc_vars)
		self.disc_solver = utils_mpi.MPImanager(self.sess, disc_trainer, disc_vars)
		# grads, params = zip(*disc_trainer.compute_gradients(loss));
		# grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
		
		# grads_and_vars = list(zip(grads, params))
		# disc_train_op = disc_trainer.apply_gradients(grads_and_vars)

		return disc_trainer, loss

	def _tf_vars(self, scope=''):
		with self.sess.as_default(), tf.Graph().as_default():
			res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope= scope)
			assert len(res) > 0
		return res

	def createActor(self, clip=True):
		actor_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
		grads, params = zip(*actor_trainer.compute_gradients(self.loss_actor));
		if clip:
			grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
		
		grads_and_vars = list(zip(grads, params))
		actor_train_op = actor_trainer.apply_gradients(grads_and_vars)

		return actor_trainer, actor_train_op

	def createCriticNetwork(self, name, input, TD, clip=True):
		critic = Critic(self.sess, name, input)
			
		with tf.variable_scope(name+'_Optimize'):
			value_loss = tf.reduce_mean(tf.square(critic.value - TD))
			loss = value_loss

		critic_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_critic)
		grads, params = zip(*critic_trainer.compute_gradients(loss))
		if clip:
			grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
		
		grads_and_vars = list(zip(grads, params))
		critic_train_op = critic_trainer.apply_gradients(grads_and_vars)
			
		return critic, critic_train_op, loss

	def buildOptimize(self, name):

		self.state = tf.placeholder(tf.float32, shape=[None, self.num_state], name=name+'_state')
		self.pose_real = tf.placeholder(tf.float32, shape=[None,self.num_feature], name=name+'_real_data')
		self.pose_sim = tf.placeholder(tf.float32, shape=[None,self.num_feature], name=name+'_fake_data')
		self.actor = Actor(self.sess, name, self.state, self.num_action)

		with tf.variable_scope(name+'_Optimize'):
			self.action = tf.placeholder(tf.float32, shape=[None,self.num_action], name='action')
			self.TD = tf.placeholder(tf.float32, shape=[None], name='TD')
			self.GAE = tf.placeholder(tf.float32, shape=[None], name='GAE')
			self.old_neglogp = tf.placeholder(tf.float32, shape=[None], name='old_neglogp')
			self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='learning_rate')
			self.cur_neglogp = self.actor.neglogp(self.action)
			self.ratio = tf.exp(self.old_neglogp-self.cur_neglogp)
			clipped_ratio = tf.clip_by_value(self.ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)

			surrogate = -tf.reduce_mean(tf.minimum(self.ratio*self.GAE, clipped_ratio*self.GAE))
			self.loss_actor = surrogate

		self.disc_trainer, self.loss_disc = self.createDiscriminator(name,self.grad_penalty,self.pose_real,self.pose_sim)

		actor_trainer, self.actor_train_op= self.createActor()

		self.critic, self.critic_train_op, self.loss_critic = self.createCriticNetwork(name, self.state, self.TD)

		var_list = tf.trainable_variables()
		save_list = []
		for v in var_list:
			if "Actor" in v.name or "Critic" in v.name or "Discriminator" in v.name:
				save_list.append(v)

		self.saver = tf.train.Saver(var_list=save_list, max_to_keep=1)
		self.sess.run(tf.global_variables_initializer())

		self.disc_solver.sync()


	def discReward(self, logits):
		r = 1.0 - 0.25 * np.square(1.0 - logits)
		r = np.maximum(r, 0.0)
		return r

	def update(self,it, tuples):
		# Discriminator Update

		# ind = np.arange(len(agent_poses))
		# np.random.shuffle(ind)
		lossval_disc = 0
		loss_agent =0
		loss_expert =0
		mean_expert =0
		mean_agent =0
		loss_grad=0

		disc_steps=1
		# update_iter = int(len(ind)//self.batch_size_disc)

		batch_size = int(np.ceil(self.batch_size_disc/self.num_procs))
		update_iter = int(self.agent_buffer.get_current_size()// self.batch_size_disc)

		assert self.agent_buffer.get_current_size() == self.expert_buffer.get_current_size()

		disc_rewards=[]
		for _ in range(disc_steps):
			for s in range(update_iter):
				agent_idx = self.agent_buffer.getRandIndex(self.batch_size_disc)
				expert_idx = self.expert_buffer.getRandIndex(self.batch_size_disc)

				agent_poses = self.agent_buffer.getSample(agent_idx)
				expert_poses = self.expert_buffer.getSample(agent_idx)

				assert not np.any(np.isnan(expert_poses))
				assert not np.any(np.isnan(agent_poses))

				grads,_, m_r,m_f,l_f,l_r,l_p,pred_f = self.sess.run(
								[self.disc_grads,self.loss_disc,	
								self.mean_real,self.mean_fake,self.fake_loss, 
								self.real_loss,self.penalty_loss, self.disc_fake.pred], 
					feed_dict={
						self.pose_real: expert_poses,
						self.pose_sim: agent_poses
					}
				)
				loss_agent+=l_f
				loss_expert +=l_r
				loss_grad += l_p
				mean_expert += m_r
				mean_agent += m_f

				self.disc_solver.update_grad(grads)

				disc_reward = self.discReward(pred_f).flatten()
				disc_rewards.append(disc_reward)
		
		
		# summary = self.sess.run(self.merged_summary)
		# global_step = tf.train.global_step()
		# self.writer.add_summary(summary, global_step=global_step)


		self.disc_info = []
		self.disc_info.append(['agent loss', loss_agent/ disc_steps])
		self.disc_info.append(['expert loss', loss_expert/ disc_steps])
		self.disc_info.append(['gradient penalty', loss_grad/disc_steps])
		self.disc_info.append(['mean agent logits', mean_agent / (update_iter*disc_steps)])
		self.disc_info.append(['mean expert logits', mean_expert / (update_iter*disc_steps)])
		self.disc_info.append(['mean discriminator reward', np.mean(disc_rewards)])
		

		# Policy Update
		state_batch, action_batch, TD_batch, neglogp_batch, GAE_batch = self.computeTDandGAE(tuples)
		GAE_batch = (GAE_batch - GAE_batch.mean())/(GAE_batch.std() + 1e-5)
		if len(state_batch) < self.batch_size:
			return


		ind = np.arange(len(state_batch))
		np.random.shuffle(ind)
		lossval_ac = 0
		lossval_c = 0
		for _ in range(5):
			for s in range(int(len(ind)//self.batch_size)):
				selectedIndex = ind[s*self.batch_size:(s+1)*self.batch_size]
				val = self.sess.run([self.actor_train_op, self.critic_train_op,
									self.loss_actor, self.loss_critic], 
					feed_dict={
						self.state: state_batch[selectedIndex], 
						self.TD: TD_batch[selectedIndex], 
						self.action: action_batch[selectedIndex], 
						self.old_neglogp: neglogp_batch[selectedIndex], 
						self.GAE: GAE_batch[selectedIndex],
						self.learning_rate_ph:self.learning_rate_actor
					}
				)

				lossval_ac += val[2]
				lossval_c += val[3]

		
		self.lossvals = []
		self.lossvals.append(['loss actor', lossval_ac / 5])
		self.lossvals.append(['loss critic', lossval_c/ 5])

		if(self.expert_buffer.get_current_size() >10000 or self.agent_buffer.get_current_size()>10000):
			self.expert_buffer.clear()
			self.agent_buffer.clear()

	def computeTDandGAE(self, tuples):
		state_batch = []
		action_batch = []
		TD_batch = []
		neglogp_batch = []
		GAE_batch = []
		timestep = 1
		start_idx =0
		for data in tuples:
			size = len(data)		
			# get values
			features, states, actions, task_rewards, values, neglogprobs = zip(*data)
			
			values = np.concatenate((values, [0]), axis=0)
			advantages = np.zeros(size)
			ad_t = 0
			
			pred_disc = self.disc_fake.getPrediction(features)
			disc_rewards = self.discReward(pred_disc).flatten()

			rewards = self.task_reward_w*np.array(task_rewards) + (1-self.task_reward_w)*disc_rewards

			for i in reversed(range(len(data))):

				t = integrate.quad(lambda x: pow(self.gamma, x), 0, timestep)[0]
				delta = t * rewards[i] + values[i+1] * pow(self.gamma, timestep)  - values[i]
				ad_t = delta + pow(self.lambd, timestep)* pow(self.gamma, timestep)  * ad_t
				advantages[i] = ad_t

			TD = values[:size] + advantages
			for i in range(size):
				state_batch.append(states[i])
				action_batch.append(actions[i])
				TD_batch.append(TD[i])
				neglogp_batch.append(neglogprobs[i])
				GAE_batch.append(advantages[i])


		return np.array(state_batch), np.array(action_batch), np.array(TD_batch), np.array(neglogp_batch), np.array(GAE_batch)



	def save(self):
		self.saver.save(self.sess, self.directory + "network", global_step = 0)
		self.env.RMS.save(self.directory+'rms-0')
		self.env.RMS_disc.save(self.directory+'disc-rms-0')


	def load(self, path):
		print("Loading parameters from {}".format(path))

		def get_tensors_in_checkpoint_file(file_name):
			varlist=[]
			var_value =[]
			reader = pywrap_tensorflow.NewCheckpointReader(file_name)
			var_to_shape_map = reader.get_variable_to_shape_map()
			for key in sorted(var_to_shape_map):
				varlist.append(key)
				var_value.append(reader.get_tensor(key))
			return (varlist, var_value)

		saved_variables, saved_values = get_tensors_in_checkpoint_file(path)

		saved_dict = {n : v for n, v in zip(saved_variables, saved_values)}
		restore_op = []
		for v in tf.trainable_variables():
			if v.name[:-2] in saved_dict:
				saved_v = saved_dict[v.name[:-2]]
				if v.shape == saved_v.shape:
					print("Restore {}".format(v.name[:-2]))
					restore_op.append(v.assign(saved_v))
				elif "L1/kernel" in v.name and v.shape[0] > saved_v.shape[0]:
					l = v.shape[0] - saved_v.shape[0]
					new_v = np.zeros((l, v.shape[1]), dtype=np.float32)
					saved_v = np.concatenate((saved_v, new_v), axis=0)
					restore_op.append(v.assign(saved_v))
					print("Restore {}, add {} input nodes".format(v.name[:-2], l))

				elif ("mean/bias" in v.name or "std" in v.name) and v.shape[0] > saved_v.shape[0]:
					l = v.shape[0] - saved_v.shape[0]
					new_v = np.zeros(l, dtype=np.float32)
					saved_v = np.concatenate((saved_v, new_v), axis=0)
					restore_op.append(v.assign(saved_v))
					print("Restore {}, add {} output nodes".format(v.name[:-2], l))

				elif "mean/kernel" in v.name and v.shape[1] > saved_v.shape[1]:
					l = v.shape[1] - saved_v.shape[1]
					new_v = np.zeros((v.shape[0], l), dtype=np.float32)
					saved_v = np.concatenate((saved_v, new_v), axis=1)
					restore_op.append(v.assign(saved_v))
					print("Restore {}, add {} output nodes".format(v.name[:-2], l))

		restore_op = tf.group(*restore_op)
		self.sess.run(restore_op)


	def printNetworkSummary(self):
		print_list = []
		print_list.append('noise : {:.3f}'.format(self.sess.run(self.actor.std).mean()))
		for v in self.lossvals:
			print_list.append('{}: {:.3f}'.format(v[0], v[1]))

		for v in self.disc_info:
			print_list.append('{}: {:.3f}'.format(v[0], v[1]))

		print_list.append('---------------------------------------------------------------')
		for s in print_list:
			print(s)


	def train(self, num_iteration):
		
		snapshot = tracemalloc.take_snapshot()
		
		it_cur = 0

		for it in range(num_iteration):
			for i in range(self.num_slaves):
				self.env.reset(i)
			states = self.env.getStates()
			local_step = 0
			last_print = 0

			epi_info_iter = []
			agent_container=[]
			expert_container = []

			epi_info = [[] for _ in range(self.num_slaves)]	

			while True:
				actions, neglogprobs = self.actor.getAction(states)
				values = self.critic.getValue(states)
				rewards, dones, params = self.env.step(actions)
				agent_pose = self.env.getAgentFeatures()
				expert_pose = self.env.getExpertFeatures()

				for j in range(self.num_slaves):
					if not self.env.getTerminated(j):
						if rewards[j] is not None:
							agent_container.append(agent_pose[j])
							expert_container.append(expert_pose[j])
							epi_info[j].append([agent_pose[j],states[j], actions[j], rewards[j], values[j], neglogprobs[j]])
							local_step += 1

						if dones[j]:
							if len(epi_info[j]) != 0:
								epi_info_iter.append(deepcopy(epi_info[j]))
							
							if local_step < self.steps_per_iteration:
								epi_info[j] = []
								features=[]
								self.env.reset(j)
							else:
								self.env.setTerminated(j)
				if local_step >= self.steps_per_iteration:
					if self.env.getAllTerminated():
						print('iter {} : {}/{}'.format(it+1, local_step, self.steps_per_iteration),end='\r')
						break
				if last_print + 100 < local_step: 
					print('iter {} : {}/{}'.format(it+1, local_step, self.steps_per_iteration),end='\r')
					last_print = local_step

				# if(len(features)!=0):
				# 	feature_container= np.concatenate((feature_container, features),axis=0)
				states = self.env.getStates()

			it_cur += 1
			print('')
			self.agent_buffer.store(agent_container)
			self.expert_buffer.store(expert_container)
			self.update(it,epi_info_iter) 

			agent_container=[]
			expert_container=[]
			epi_info_iter = []

			if self.learning_rate_actor > 1e-5:
				self.learning_rate_actor = self.learning_rate_actor * self.learning_rate_decay

			
			if it_cur % 5 == 4:
				summary = self.env.printSummary()
				self.printNetworkSummary()
				if self.directory is not None:
					self.save()

				if self.directory is not None and self.reward_max < summary['r_per_e']:
					self.reward_max = summary['r_per_e']
					self.env.RMS.save(self.directory+'rms-rmax')
					self.env.RMS_disc.save(self.directory+'disc-rms-rmax')

					os.system("cp {}/network-{}.data-00000-of-00001 {}/network-rmax.data-00000-of-00001".format(self.directory, 0, self.directory))
					os.system("cp {}/network-{}.index {}/network-rmax.index".format(self.directory, 0, self.directory))
					os.system("cp {}/network-{}.meta {}/network-rmax.meta".format(self.directory, 0, self.directory))

			if it_cur % 100 == 1:

				snapshot2 = tracemalloc.take_snapshot()
				top_stats = snapshot2.compare_to(snapshot,'lineno')
				print("[Top 10 differences]")
				for stat in top_stats[:10]:
					print(stat)

	def run(self, state):
		state = np.reshape(state, (1, self.num_state))
		state = self.RMS.apply(state)
		
		values = self.critic.getValue(state)
		# action, _ = self.actor.getAction(state)
		action = self.actor.getMeanAction(state)

		return action

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--ntimesteps", type=int, default=1000000)
	parser.add_argument("--ref", type=str, default="")
	parser.add_argument("--test_name", type=str, default="")
	parser.add_argument("--pretrain", type=str, default="")
	parser.add_argument("--nslaves", type=int, default=4)
	parser.add_argument("--save", type=bool, default=True)
	parser.add_argument("--no-plot", dest='plot', action='store_false')
	parser.set_defaults(plot=True)

	args = parser.parse_args()

	directory = None
	if args.save:
		if not os.path.exists("./output/"):
			os.mkdir("./output/")

		directory = "./output/" + args.test_name + "/"
		if not os.path.exists(directory):
			os.mkdir(directory)
	if args.pretrain != "":
		env = Monitor(ref=args.ref, num_slaves=args.nslaves, directory=directory, plot=args.plot)
	else:
		env = Monitor(ref=args.ref, num_slaves=args.nslaves, directory=directory, plot=args.plot)

	tracemalloc.start()
	amp = AMP()
	amp.initTrain(env=env, name=args.test_name, directory=directory, pretrain=args.pretrain)
	amp.train(args.ntimesteps)