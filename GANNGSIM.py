import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

se = 1234
np.random.seed(se)
tf.set_random_seed(se)
u_max = 80
rho_max = 0.12
sigma = 0.1
lamda = 0.01
datarho = pd.read_csv('//home/naman//MFG//NGSIM_US101_Density_Data.csv').values.T
datau = pd.read_csv('//home//naman//MFG//NGSIM_US101_Velocity_Data.csv').values.T

dataV = np.zeros((540, 104, 104))
dataV[-1, :, :] = 0

uin = u_max*(1-datarho[0]/rho_max)
dataV[:, 0, :] = 1/uin

datarho = datarho/rho_max
datau = datau/u_max
u_max = 1 #rescaled
rho_max = 1 #rescaled

dataomega = datau+datarho*u_max/rho_max
dataomega = dataomega



datautilde = np.zeros((540, 104, 104))

for j in range(104):
	datautilde[:, :, j] = datau

uxn = 104
xlo = 0
xhi = 2060

utn = 540
tlo = 0
thi = 2695

uwn = 104
wlo = 0
whi = 1

Nce = uxn+utn*2
Nhjb = uwn*uxn
Nf = 25000
x = np.linspace(xlo, xhi, uxn)
t = np.linspace(tlo, thi, utn)
w = np.linspace(wlo, whi, uwn)
dataV[-1] = 0
XCE, TCE = np.meshgrid(x, t)
XHJB, THJB, WHJB = np.meshgrid(x, t, w)

inputCE = np.hstack([XCE.flatten()[:, None], TCE.flatten()[:, None]])
inputHJB = np.hstack([XHJB.flatten()[:, None], THJB.flatten()[:, None], WHJB.flatten()[:, None]])

rho_star = datarho.flatten()[:, None]
u_star = datau.flatten()[:, None]
omega_star = dataomega.flatten()[:, None]
utildestar = datautilde.flatten()[:, None]
V_star = dataV.flatten()[:, None]

xce0 = np.hstack([XCE[0:1, :].reshape((uxn,1)), TCE[0:1, :].reshape((uxn,1))])
xce1 = np.hstack([XCE[:, 0:1].reshape((utn, 1)), TCE[:, 0:1].reshape((utn, 1))])
xce2 = np.hstack([XCE[:, -1:].reshape((utn, 1)), TCE[:, -1:].reshape((utn, 1))])
xce3 = np.hstack((XCE[:, 15:16], TCE[:, 15:16]))
xce4 = np.hstack((XCE[:, 90:91], TCE[:, 90:91]))
xce5 = np.hstack((XCE[:, 25:26], TCE[:, 25:26]))
xce6 = np.hstack((XCE[:, 75:76], TCE[:, 75:76]))

rho0 = datarho[0:1, :].reshape((uxn, 1)) #Initial condition for the CE equation
rho1 = datarho[:, 0:1].reshape((utn, 1)) #Boundary condition for the location, CE
rho2 = datarho[:, -1:].reshape((utn, 1)) #Boundary condition for the location, CE
rho3 = datarho[:, 15:16].reshape((utn, 1))
rho4 = datarho[:, 90:91].reshape((utn, 1))
rho5 = datarho[:, 25:26].reshape((utn, 1))
rho6 = datarho[:, 75:76].reshape((utn, 1))

omega0 = dataomega[0:1, :].reshape((uxn, 1)) #Initial condition for the CE equation
omega1 = dataomega[:, 0:1].reshape((utn, 1)) #Boundary condition for the location, CE
omega2 = dataomega[:, -1:].reshape((utn, 1)) #Boundary condition for the location, CE
omega3 = dataomega[:, 15:16].reshape((utn, 1))
omega4 = dataomega[:, 90:91].reshape((utn, 1))
omega5 = dataomega[:, 25:26].reshape((utn, 1))
omega6 = dataomega[:, 75:76].reshape((utn, 1))

u0 = datau[0:1, :].reshape((uxn, 1)) # Initial condition for the CE equation
u1 = datau[:, 0:1].reshape((utn, 1)) #Boundary condition for the location, CE
u2 = datau[:, -1:].reshape((utn, 1)) #Boundary condition for the location, CE
u3 = datau[:, 15:16].reshape((utn, 1))
u4 = datau[:, 90:91].reshape((utn, 1))
u5 = datau[:, 25:26].reshape((utn, 1))
u6 = datau[:, 75:76].reshape((utn, 1))


xhjb0 = np.hstack([XHJB[-1:, :, :].reshape((uxn*uwn,1)), THJB[-1:, :, :].reshape((uxn*uwn,1)), WHJB[-1:, :, :].reshape((uxn*uwn,1))]) #Terminal condition for the HJB equation
xhjb1 = np.hstack([XHJB[:, 0:1, :].reshape((utn*uwn, 1)), THJB[:, 0:1, :].reshape((utn*uwn, 1)),  WHJB[:, 0:1, :].reshape((utn*uwn, 1))]) #Lower BC spatial points for the HJB equation
xhjb2 = np.hstack([XHJB[:, -1:, :].reshape((utn*uwn, 1)), THJB[:, -1:, :].reshape((utn*uwn, 1)),  WHJB[:, -1:, :].reshape((utn*uwn, 1))]) #Terminal BC spatial points for the HJB equation
xhjb3 = np.hstack([XHJB[:, :, 0:1].reshape((utn*uwn, 1)), THJB[:, :, 0:1].reshape((utn*uwn, 1)),  WHJB[:, :, 0:1].reshape((utn*uwn, 1))]) #Lower BC property points for the HJB equation
xhjb4 = np.hstack([XHJB[:, :, -1:].reshape((utn*uwn, 1)), THJB[:, :, -1:].reshape((utn*uwn, 1)),  WHJB[:, :, -1:].reshape((utn*uwn, 1))]) #Terminal BC property points for the HJB equation



V0 = dataV[-1:, :, :].reshape((uxn*uwn, 1)) # Terminal condition for the HJB equation
V1 = dataV[:, 0:1, :].reshape((utn*uwn, 1))	#Boundary condition for the location, HJB
V2 = dataV[:, -1:, :].reshape((utn*uwn, 1))	#Boundary condition for the location, HJB
V3 = dataV[:, :, 0:1].reshape((utn*uxn, 1))	#Boundary condition for the property, HJB
V4 = dataV[:, :, -1:].reshape((utn*uxn, 1))	#Boundary condition for the property, HJB
utilde0 = datautilde[-1:, :, :].reshape((uxn*uwn, 1)) # Terminal condition for the HJB equation
utilde1 = datautilde[:, 0:1, :].reshape((utn*uwn, 1))	#Boundary condition for the location, utilde
utilde2 = datautilde[:, -1:, :].reshape((utn*uwn, 1))	#Boundary condition for the location, utilde
utilde3 = datautilde[:, :, 0:1].reshape((utn*uxn, 1))	#Boundary condition for the property, utilde
utilde4 = datautilde[:, :, -1:].reshape((utn*uxn, 1))	#Boundary condition for the property, utilde

X_CE_train1 = np.vstack([xce0, xce1, xce2])#, xce3, xce4, xce5, xce6])
rho_train1 = np.vstack([rho0, rho1, rho2])#, rho3])#, rho4, rho5, rho6])
omega_train1 = np.vstack([omega0, omega1, omega2, omega3, omega4, omega5, omega6])
u_train1 = np.vstack([u0, u1, u2])#, u3, u4, u5, u6])

X_HJB_train1 = np.vstack([xhjb0, xhjb1, xhjb2, xhjb3, xhjb4])
V_train1 = np.vstack([V0, V1, V2, V3, V4])
utilde_train1 = np.vstack([utilde0, utilde1, utilde2, utilde3, utilde4])

idx1 = np.random.choice(X_CE_train1.shape[0], Nce, replace=False)
idx2 = np.random.choice(X_HJB_train1.shape[0], Nhjb, replace = False)
X_CE_train = X_CE_train1[idx1, :]
u_train = u_train1[idx1, :]
rho_train = rho_train1[idx1,:]
omega_train = omega_train1[idx1,:]

X_HJB_train = X_HJB_train1[idx2, :]
V_train = V_train1[idx2, :]
utilde_train = utilde_train1[idx2, :]

lbce = inputCE.min(0)
ubce = inputCE.max(0)
lbhjb = inputHJB.min(0)
ubhjb = inputHJB.max(0)
aux = lbhjb + (ubhjb - lbhjb) * lhs(3, Nf)


class MFG_PIGAN:
	def __init__(self, X_CE, rho, u, omega, X_HJB, V, utilde, aux, lbce, ubce, lbhjb, ubhjb, layersg, layersu, layersd):
		self.lbce = lbce
		self.ubce = ubce
		self.lbhjb = lbhjb
		self.ubhjb = ubhjb

		self.xce = X_CE[:, 0:1]
		self.tce = X_CE[:, 1:2]

		self.xhjb = X_HJB[:, 0:1]
		self.thjb = X_HJB[:, 1:2]
		self.whjb = X_HJB[:, 2:3]

		self.Ax = aux[:, 0:1]
		self.At = aux[:, 1:2]
		self.Aw = aux[:, 2:3]

		self.u = u
		self.rho = rho
		self.omega = omega
		self.V = V
		self.utilde = utilde

		self.layersg = layersg
		self.layersu = layersu
		self.layersd = layersd
		self.WCE, self.BCE = self.initialise_generator(layersg)
		self.WHJB, self.BHJB = self.initialise_discriminator(layersd)
		self.Wu, self.Bu = self.initialise_u_net(layersu)
		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

		self.lr = tf.placeholder(tf.float32, shape=[])
		self.xcep = tf.placeholder(tf.float32, shape=[None, self.xce.shape[1]])
		self.tcep = tf.placeholder(tf.float32, shape=[None, self.tce.shape[1]])
		self.xhjbp = tf.placeholder(tf.float32, shape=[None, self.xhjb.shape[1]])
		self.thjbp = tf.placeholder(tf.float32, shape=[None, self.thjb.shape[1]])
		self.whjbp = tf.placeholder(tf.float32, shape=[None, self.whjb.shape[1]])
		self.up   = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
		self.rhop = tf.placeholder(tf.float32, shape=[None, self.rho.shape[1]])
		self.omegap = tf.placeholder(tf.float32, shape=[None, self.omega.shape[1]])
		self.Vp = tf.placeholder(tf.float32, shape=[None, self.V.shape[1]])
		self.utildep = tf.placeholder(tf.float32, shape=[None, self.utilde.shape[1]])

		self.Axp = tf.placeholder(tf.float32, shape=[None, self.Ax.shape[1]])
		self.Atp = tf.placeholder(tf.float32, shape=[None, self.At.shape[1]])
		self.Awp = tf.placeholder(tf.float32, shape=[None, self.Aw.shape[1]])

		self.Ygen = self.generator(self.xcep, self.tcep)
		self.Ydis = self.discriminator(self.xhjbp, self.thjbp, self.whjbp)
		self.rho_pred = self.Ygen[:, 0:1]
		self.omega_pred = self.Ygen[:, 1:2]
		self.V_pred = self.Ydis[:, 0:1]
		self.CE_res = self.CE(self.Axp, self.Atp)
		self.ce_res1 = self.CE_res[0]
		self.ce_res2 = self.CE_res[1]
		self.f = self.HJB(self.Axp, self.Atp, self.Awp)
		self.hjb_res = self.f[0]
		self.utilde_res = self.f[1]
		self.u_res = self.f[2]
		self.f_pred = self.HJB(self.xhjbp, self.thjbp, self.whjbp)
		self.utilde_pred = self.f_pred[1]
		self.u_pred = self.net_u(self.xcep, self.tcep)
		self.u_res_pred = self.net_u(self.Axp, self.Atp)

		self.gen_loss = 100*tf.reduce_mean(tf.square(self.rhop-self.rho_pred))+100*tf.reduce_mean(tf.square(self.omegap-self.omega_pred))+tf.reduce_mean(tf.square(self.ce_res1))+tf.reduce_mean(tf.square(self.ce_res2))
		self.dis_loss = tf.reduce_mean(tf.square(self.Vp-self.V_pred))+tf.reduce_mean(tf.square(self.hjb_res))+tf.reduce_mean(tf.square(self.utildep-self.utilde_pred))
		self.u_loss = 100*tf.reduce_mean(tf.square(self.up-self.u_pred))+tf.reduce_mean(tf.square(self.u_res_pred-self.u_res))
		
		self.gen_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.999).minimize(self.gen_loss)
		self.dis_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.999).minimize(self.dis_loss)
		self.d_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.dis_loss, method = 'L-BFGS-B', options = {'maxiter': 1000,'maxfun': 50000,'maxcor': 50,'maxls': 50,'ftol' :1.0 * np.finfo(float).eps})
		self.g_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.gen_loss, method = 'L-BFGS-B', options = {'maxiter': 1000,'maxfun': 50000,'maxcor': 50,'maxls': 50,'ftol' :1.0 * np.finfo(float).eps})
		self.u_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.u_loss, method = 'L-BFGS-B', options = {'maxiter': 1000,'maxfun': 50000,'maxcor': 50,'maxls': 50,'ftol' :1.0 * np.finfo(float).eps})

		init = tf.global_variables_initializer()
		self.sess.run(init)

	def initialise_u_net(self, layers):
		Wu = []
		Bu = []
		for l in range(len(layers) - 1):
			W = tf.Variable(tf.truncated_normal([layers[l], layers[l + 1]], mean=0, stddev=0.1), dtype=tf.float32)
			b = tf.Variable(tf.zeros([layers[l + 1]]), dtype=tf.float32)
			Wu.append(W)
			Bu.append(b)
		return Wu, Bu

	def initialise_generator(self, layers):
		Wg = []
		Bg = []
		for l in range(len(layers) - 1):
			W = tf.Variable(tf.truncated_normal([layers[l], layers[l + 1]], mean=0, stddev=0.1), dtype=tf.float32)
			b = tf.Variable(tf.zeros([layers[l + 1]]), dtype=tf.float32)
			Wg.append(W)
			Bg.append(b)
		return Wg, Bg

	def initialise_discriminator(self, layers):
		Wd = []
		Bd = []
		for l in range(len(layers) - 1):
			W = tf.Variable(tf.truncated_normal([layers[l], layers[l + 1]], mean=0, stddev=0.1), dtype=tf.float32)
			b = tf.Variable(tf.zeros([layers[l + 1]]), dtype=tf.float32)
			Wd.append(W)
			Bd.append(b)
		return Wd, Bd

	def velocity(self, X, Wu, Bu):
		L = len(Wu) + 1
		A = 2*(X-self.lbce)/(self.ubce-self.lbce)-1
		for l in range(L-2):
			W = Wu[l]
			b = Bu[l]
			Z = tf.add(tf.matmul(A, W), b)
			A = tf.tanh(Z)
		W = Wu[-1]
		b = Bu[-1]
		Z = tf.add(tf.matmul(A, W), b)
		Y = Z
		return Y

	def gen(self, X, WCE, BCE):
		L = len(WCE) + 1
		A = 2*(X-self.lbce)/(self.ubce-self.lbce)-1
		for l in range(L-2):
			W = WCE[l]
			b = BCE[l]
			Z = tf.add(tf.matmul(A, W), b)
			A = tf.tanh(Z)
		W = WCE[-1]
		b = BCE[-1]
		Z = tf.add(tf.matmul(A, W), b)
		Y = Z
		return Y

	def dis(self, X, WHJB, BHJB):
		L = len(WHJB) + 1
		A = 2*(X-self.lbhjb)/(self.ubhjb-self.lbhjb)-1
		for l in range(L-2):
			W = WHJB[l]
			b = BHJB[l]
			Z = tf.add(tf.matmul(A, W), b)
			A = tf.tanh(Z)
		W = WHJB[-1]
		b = BHJB[-1]
		Z = tf.add(tf.matmul(A, W), b)
		Y = Z
		return Y

	def generator(self, x, t):
		Ygen = self.gen(tf.concat([x, t], 1), self.WCE, self.BCE)
		return Ygen

	def discriminator(self, x, t, w):
		Ydis = self.dis(tf.concat([x, t, w], 1), self.WHJB, self.BHJB)
		return Ydis
	
	def net_u(self, x, t):
		u = self.velocity(tf.concat([x, t], 1), self.Wu, self.Bu)
		return u

	def CE(self, x, t):#Call this function on the auxilliary points
	 	NN = self.generator(x, t) #x and t components of the auxilliary points
	 	u = self.net_u(x, t) #x and t components of the auxilliary points
	 	rho = NN[:, 0:1]
	 	omega = NN[:, 1:2]
	 	drho_t = tf.gradients(rho,t)[0]
	 	drho_x = tf.gradients(rho, x)[0]
	 	# drho_xx = tf.gradients(drho_x, x)[0]
	 	drhou_x = tf.gradients(rho*u, x)[0]
	 	dz_t = tf.gradients(rho*omega, t)[0]
	 	dzu_x = tf.gradients(rho*omega*u, x)[0]
	 	U_eq = u_max*(omega/u_max-rho/rho_max)
	 	f1 = drho_t+drhou_x#-0.01*drho_xx
	 	f2 = dz_t+dzu_x-rho*(lamda*(U_eq-u))
	 	return f1, f2
	
	def HJB(self, x, t, w):
	 	Y = self.generator(x, t) #Call this function on the x and t components of the auxilliary points
	 	rho = Y[:, 0:1]
	 	omega = Y[:, 1:2]
	 	V = self.discriminator(x, t, w) #Call this function on the auxilliary points
	 	dV_t = tf.gradients(V, t)[0]
	 	dV_x = tf.gradients(V, x)[0]
	 	dV_w = tf.gradients(V, w)[0]
	 	dV_ww = tf.gradients(dV_w, w)[0]
	 	U_eq = u_max*(omega/u_max-rho/rho_max)
	 	Theta = (dV_x-lamda*dV_w)
	 	# f = dV_t+U_eq*dV_x-1/2*(dV_x-lamda*dV_w)**2-2*lamda*U_eq*dV_w+sigma**2*(1-rho)*dV_ww #viscous ARZ-MFG
	 	# utilde = U_eq-dV_x+lamda*dV_w
	 	f = dV_t+U_eq*Theta+Theta**2*(0.5-u_max**2)+0.5*(1-omega/u_max)**2+0.5*(1-rho/rho_max)-2*lamda*U_eq*dV_w+sigma**2*(1-rho)*dV_ww#viscous non-sep
	 	utilde = U_eq-u_max**2*Theta
	 	dutilde_w = tf.gradients(utilde, w)[0]
	 	dutilde_ww = tf.gradients(dutilde_w, w)[0]
	 	u = utilde+(1-2*w)/2*dutilde_w+(1-3*w+3*w**2)/6*dutilde_ww #call the net_u function on the auxilliary points and evaluate net_u-u error	 		 	
	 	return f, utilde, u

	def train_GAN(self, lr, epochs):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in range(epochs):
				if (epoch+1)%100 == 0:
					lr = lr/10
				sess.run(self.dis_optimizer, feed_dict={self.lr: lr, self.xhjbp: self.xhjb, self.thjbp: self.thjb, self.whjbp: self.whjb, self.Vp: self.V,
					self.Axp: self.Ax, self.Atp: self.At, self.Awp: self.Aw, self.xcep: self.xce, self.wcep: self.wce, self.tcep: self.tce})
				sess.run(self.gen_optimizer, feed_dict={self.lr: lr, self.xcep: self.xce, self.tcep: self.tce, self.wcep: self.wce, self.rhop: self.rho, self.omegap: self.omega,
					self.up:self.u, self.rhop: self.rho, self.Axp: self.Ax, self.Atp: self.At, self.Awp: self.Aw})
				d_loss_val = sess.run(self.dis_loss, feed_dict={self.xhjbp: self.xhjb, self.thjbp: self.thjb, self.whjbp: self.whjb, self.Vp: self.V, self.utildep: self.utilde,
							self.Axp: self.Ax, self.Atp: self.At, self.Awp: self.Aw, self.xcep: self.xce, self.wcep: self.wce, self.tcep: self.tce})
				g_loss_val = sess.run(self.gen_loss, feed_dict={self.xcep: self.xce, self.tcep: self.tce, self.wcep: self.wce, self.rhop: self.rho, self.omegap: self.omega,
						self.up:self.u, self.rhop: self.rho, self.Axp: self.Ax, self.Atp: self.At, self.Awp: self.Aw})
				print(f'Epoch {epoch}: HJB Loss = {d_loss_val}, CE Loss = {g_loss_val}')
	
	
	def callback(self, loss):
		print('Loss:', loss)

	def train_GAN_lbfgsb(self, epochs):
		feed_dict1 = {self.xhjbp: self.xhjb, self.thjbp: self.thjb, self.whjbp: self.whjb, self.Vp: self.V, self.utildep: self.utilde,
					self.Axp: self.Ax, self.Atp: self.At, self.Awp: self.Aw,}
		feed_dict2 = feed_dict={self.xcep: self.xce, self.tcep: self.tce, self.rhop: self.rho, self.omegap: self.omega,
					self.up: self.u, self.Axp: self.Ax, self.Atp: self.At}
		feed_dict3 = {self.xcep: self.xce, self.tcep: self.tce, self.up: self.u, self.Axp: self.Ax, self.Atp: self.At, self.Awp:self.Aw}
		for epoch in range(epochs):
			print('Training the HJB Equation:...')
			self.d_optimizer.minimize(self.sess, feed_dict = feed_dict1, fetches = [self.dis_loss], loss_callback = self.callback)
			print('Training the Continuity Equation:...')
			self.g_optimizer.minimize(self.sess, feed_dict = feed_dict2, fetches = [self.gen_loss], loss_callback = self.callback)
			print('Training the Mean Field Velocity:...')
			self.u_optimizer.minimize(self.sess, feed_dict = feed_dict3, fetches = [self.u_loss], loss_callback = self.callback)
	
	def predict(self, inputCE, inputHJB):
		Y_star = self.sess.run(self.Ygen, {self.xcep: inputCE[:, 0:1], self.tcep: inputCE[:, 1:2]})
		V_star = self.sess.run(self.Ydis, {self.xhjbp: inputHJB[:,0:1], self.thjbp: inputHJB[:, 1:2], self.whjbp: inputHJB[:, 2:3]})
		U_star = self.sess.run(self.u_pred, {self.xcep: inputCE[:, 0:1], self.tcep: inputCE[:, 1:2]})
		return Y_star, V_star, U_star

layersg = [2, 100, 100, 100, 100, 100, 100, 2]
layersd = [3, 100, 100, 100, 100, 100, 100, 1]
layersu = [2, 100, 100, 100, 100, 100, 100, 1]
model = MFG_PIGAN(X_CE_train, rho_train, u_train, omega_train, X_HJB_train, V_train, utilde_train, aux, lbce, ubce, lbhjb, ubhjb, layersg, layersu, layersd)
initial_time = time.time()
model.train_GAN_lbfgsb(epochs = 5)
elapsed = time.time() - initial_time
print('Training time: %.4f seconds' % elapsed)
Y_pred, V_pred, U_pred = model.predict(inputCE, inputHJB)
rho_pred = Y_pred[:, 0:1]
omega_pred = Y_pred[:, 1:2]
pd.DataFrame(U_pred).to_csv('//home//naman//MFG//U_Predict_coarseNGSIMnonsep.csv', index = False)
pd.DataFrame(omega_pred).to_csv('//home//naman//MFG//omega_Predict_coarseNGSIMnonsep.csv', index = False)
pd.DataFrame(rho_pred).to_csv('//home//naman//MFG//rho_Predict_coarseNGSIMnonsep.csv', index = False)