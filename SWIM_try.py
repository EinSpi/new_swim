import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import swim_backbones.dense
import swim_backbones.linear
import swim_model
import activations.activations as act
from utils.data_prepare import load_training_data_from_mat
import numpy as np
from scipy.interpolate import griddata
from utils.plotting import newfig, savefig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def l2_error_relative(f_approx, f_true):
		return np.linalg.norm(f_approx-f_true, ord=2) / np.linalg.norm(f_true, ord=2)

def mse(f_approx,f_true):
		return np.mean((f_approx-f_true)**2)

#data
data_path="Data\\discontinuous_complicated.mat"

X_train, u_train, X_val, u_val, X_idn_star, u_idn_star, T_idn, X_idn, Exact_idn=load_training_data_from_mat(data_path=data_path, seed=42)
lb_idn = np.array([0.0, -20.0])
ub_idn = np.array([40.0, 20.0])
keep=1
#model
#activation=act.Rational(num_coeff_p=3,num_coeff_q=2)
activation=act.Rational(num_coeff_p=4,num_coeff_q=3)
dense=swim_backbones.dense.Dense(layer_width=800,activation=activation,random_seed=42,repetition_scaler=4,set_size=7)
linear=swim_backbones.linear.Linear()
model=swim_model.Swim_Model([dense,linear])

#train
model.fit(X=X_train,y=u_train)

#print(dense.weights)
#infer
u_pred_identifier=model(X_idn_star)
#compute loss
u_pred_identifier=u_pred_identifier.detach().cpu().numpy()#torch to numpy
mean_squared_error=mse(u_pred_identifier,u_idn_star)
rel_l2_error=l2_error_relative(u_pred_identifier,u_idn_star)
print(f"mse: {mean_squared_error}")
print(f"rel l2: {rel_l2_error}")
U_pred = griddata(X_idn_star, u_pred_identifier.flatten(), (T_idn, X_idn), method='cubic')


######################################################################
############################# Plotting ###############################
######################################################################

fig, ax = newfig(1.0, 0.6)
ax.axis('off')

######## Exact solution #######################
########      Predicted p(t,x,y)     ###########
gs = gridspec.GridSpec(1, 3)
gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.5)
ax = plt.subplot(gs[:, 0])
h = ax.imshow(Exact_idn, interpolation='nearest', cmap='jet',
                extent=[lb_idn[0], ub_idn[0]*keep, lb_idn[1], ub_idn[1]],
                origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

fig.colorbar(h, cax=cax)
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_title('Exact Dynamics', fontsize = 10)

######## Approximate ###########
ax = plt.subplot(gs[:, 1])
h = ax.imshow(U_pred, interpolation='nearest', cmap='jet',
                extent=[lb_idn[0], ub_idn[0]*keep, lb_idn[1], ub_idn[1]],
                origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

fig.colorbar(h, cax=cax)
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_title('Predicted' , fontsize = 10)

######## Approximation Error ###########
ax = plt.subplot(gs[:, 2])
h = ax.imshow(np.abs(Exact_idn-U_pred), interpolation='nearest', cmap='jet',
                extent=[lb_idn[0], ub_idn[0] * keep, lb_idn[1], ub_idn[1]],
                origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

fig.colorbar(h, cax=cax)
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_title('Error', fontsize=10)

savefig("Results/"+"dynamics")
