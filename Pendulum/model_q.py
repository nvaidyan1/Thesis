import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
# import the OrderedDict
from collections import OrderedDict
import nengo

class Get_Params(object):
    def __init__(self):
        self.seed = 42

        # Pendulum parameters
        self.m  = 1.0
        self.l  = 1.0
        self.dt = 1e-3
        self.g  = 10.0
        self.b  = 1.0

        # Target parameters
        self.q_thresh  = 0.05
        self.dq_thresh = 0.1
        self.avg_vel   = 1
        self.wait_time = 0.1
        self.reach_slow = False

        # Scaling parameters
        self.max_u = 20
        self.max_q = np.pi/2
        self.max_dq = self.avg_vel*2
        self.GTC = True

        # Controller parameters
        self.start_after = 0.2
        self.Kp = 25
        self.Kd = 25

        self.debug = True
        self.tf = 5
        self.no_obs_from = 100

        # Params for estimation
        self.Zdim = 2
        self.Udim = 1

        self.theta_s = 0.1
        self.qs = 5
        self.theta_u = 0.1
        self.qu = 5

        self.recurr = True
        self.Lr = 1e-4

        self.n_neurons = 1500
        self.Xw = np.zeros((1, self.n_neurons))
        self.save_dir = "C:\\Users\\Nat\\Documents\\Code\\Thesis\\pend\\data_saves\\"
        

    def __str__(self):
        return '\n'.join(['%s: %s' % item for item in vars(self).items()])



class SPend(object):
    def __init__(self, params):
        # sign convention: = q=0 --> down, counter-clockwise : positive
        self.m = params.m
        self.l = params.l
        self.dt = dt = params.dt
        self.g = params.g
        self.b = params.b
        self.J = self.m*self.l**2
        print('Pendulum parameters: \n',
              'm = ', self.m, '\n',
              'l = ', self.l, '\n',
              'g = ', self.g, '\n',
              'b = ', self.b, '\n',)
        
        
        self.max_q = params.max_q
        self.max_dq = params.max_dq
        self.max_u = params.max_u
        self.q_des   = 0
        self.dq_des  = 0
        self.ddq_des = 0
        self.xi      = 0 
        self.ddq     = 0
        self.reset(params.seed)
        self.cart(0)

    def reset(self, seed):
        self.rng = np.random.RandomState(seed)
        self.q = self.rng.uniform(-self.max_q, self.max_q)
        self.dq = self.rng.uniform(-self.max_dq, self.max_dq)
        print(f'Initial conditions: \n q = {self.q} \n dq = {self.dq}')
        
    def step(self, u):
        u = np.clip(u, -self.max_u, self.max_u)
        self.ddq = (u - self.b * self.dq* np.abs(self.dq) - self.m*self.g*self.l*np.sin(self.q) )/self.J
        self.dq += self.ddq*self.dt
        self.dq = np.clip(self.dq, -self.max_dq, self.max_dq)

        self.q += self.dq*self.dt
        self.q = np.clip(self.q, -self.max_q, self.max_q)
        self.cart(u)
        
    def cart(self, u):
        self.x = np.sin(self.q)
        self.y = -np.cos(self.q)
        self.dx = self.dq*np.cos(self.q)
        self.dy = self.dq*np.sin(self.q)

        # Scale using min and max values of cart vel
        self.q_scaled = self.q/self.max_q
        self.dq_scaled = self.dq/self.max_dq
        self.u_scaled = u/self.max_u



class targetman(object):
    def __init__(self, params):
        self.trials = 0
        seed = params.seed
        self.reset(seed)

        self.qf = np.array(np.pi/4)
        self.q_thresh = params.q_thresh #position thresh
        self.dq_thresh = params.dq_thresh #velocity thresh
        self.rt = 0 
        self.avg_time = 1
        self.avg_vel = params.avg_vel
        self.wait_time = params.wait_time
        self.reach_slow = params.reach_slow
        self.reach_slow_factor = 0.1 if params.reach_slow else 1
        self.hits = 0

        self.max_q = params.max_q
        self.dt = params.dt
        self.first_wait = True
        self.first_wait_time = 0.2

        self.trial = 0
        self.q_des, self.dq_des, self.ddq_des = 0, 0, 0
        
    def reset(self, train_seed=1, test_seed=1):
        self.rng_train = np.random.RandomState(train_seed) 
        self.rng_test = np.random.RandomState(test_seed)
        self.qf = np.array(np.pi/4)
        self.qi = 0
        

    def minjt(self, xi, xf, t, tf):
        t = np.clip(t, 0, tf)
        r = (t/tf)
        p_minj = xi + (xf - xi) * (10*r**3 - 15*r**4 + 6*r**5)
        v_minj = (1/tf)*(xf- xi) * (30*r**2 -60*r**3 + 30*r**4) 
        a_minj = (1/tf)**2*(xf - xi)*(60*r - 180*r**2 + 120*r**3)

        return p_minj, v_minj, a_minj        


    def step(self, q, dq):
        self.rt += self.dt
        
        # No targets for first `first_wait` seconds
        if self.first_wait and (self.rt <= self.first_wait_time):
            self.qi = q
            self.qf = q
        elif self.first_wait and (self.rt > self.first_wait_time):
            print('First wait over')
            self.first_wait = False
            self.qf = np.array(np.pi/4)
            self.rt = 0

        dist_to_target = np.abs(q - self.qf)
        vel_to_target = np.abs(dq)
        time_out = self.rt > (self.avg_time + self.wait_time)
        target_hit = (dist_to_target < self.q_thresh and vel_to_target < self.dq_thresh)
        
        if (target_hit or time_out) and not self.first_wait:
            
            if target_hit and not time_out:
                print("Hit")
                self.hits += 1
                # Increase by reach_slow_factor until it is 1
                self.reach_slow_factor = min(self.reach_slow_factor + 0.1, 1)

            self.rt = 0
            self.qi = q
            self.trial += 1
            self.qf = self.rng_train.uniform(-self.max_q, self.max_q)*self.reach_slow_factor
            self.avg_time = max(np.abs(self.qf - self.qi)/self.avg_vel, 0.5)
            
        self.q_des, self.dq_des, self.ddq_des = self.minjt(self.qi, self.qf, self.rt, self.avg_time) #All reaches in 1s

class Controller(object):
    def __init__(self, params):
        self.Kp, self.Kd = params.Kp, params.Kd

        self.m = params.m
        self.l = params.l
        self.g = params.g
        self.J = self.m*self.l**2

    def get_control(self, q, dq, q_des, dq_des):
        u = self.Kp*(q_des - q) + self.Kd*( - dq) + (self.m * self.g * self.l * np.sin(q))/self.J
        
        # u = self.Kp*(q_des - q) + self.Kd*(dq_des - dq) + (self.m * self.g * self.l * np.sin(q))/self.J
        return u

### Simulation functions

def Num_sim_pend(params, pen, targ, ctrl, tf):
    # Seed numpy
    np.random.seed(params.seed); print("Seeded with: ", params.seed)
    dt = params.dt
    time = np.arange(0, tf, dt)
    n_steps = len(time)
    dim = 2
    lamb = params.Kd/params.Kp
    k = params.Kp
    # a = 0*np.array([[params.J, params.b, params.m*params.g*params.l]]).T

    # Accumulators
    Q    = np.zeros((n_steps, dim)) # Ang pos and vel
    U    = np.zeros((n_steps, 1))   # Control
    Targ = np.zeros((n_steps, 1))   # Target angle
    Minj = np.zeros((n_steps, 2))   # Min_jerk position and velocity

    # Initial conditions
    Q[0] = pen.q, pen.dq

    for i in range(n_steps):
        #Get tartget
        targ.step(pen.q, pen.dq)
        
        
        #Controller
        u = ctrl.get_control(pen.q, pen.dq, targ.q_des, 0)
        pen.step(u)

        Targ[i] = targ.qf
        Minj[i] = targ.q_des, targ.dq_des
        Q[i] = pen.q, pen.dq
        U[i] = u #pen.u_scaled
        
    
    return (time, Q, U, Targ, Minj)



from scipy.special import legendre

def make_lmu(q=6, theta=1.0):
    Q = np.arange(q, dtype=np.float64)
    R = (2*Q + 1)[:, None] / theta
    j, i = np.meshgrid(Q, Q)

    A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
    B = (-1.)**Q[:, None] * R
    return A, B

def get_weights_for_delays(r, q=6):
    # compute the weights needed to extract the value at time r
    # from the network (r=0 is right now, r=1 is theta seconds ago)
    r = np.asarray(r)
    m = np.asarray([legendre(i)(2*r - 1) for i in range(q)])
    return m.reshape(q,-1).T


class PendNet(nengo.Network):
     def __init__(self,  params, GTC,  **kwargs):
        super().__init__(**kwargs)
    
        self.env = SPend(params)
        self.tm = targetman(params)
        self.unode = Controller(params)
        
        with self:
            def env_step(t, u):
                self.env.step(u[0])
                if t%25 == 0:
                    print(f"Time : {t}")
                return (self.env.q, self.env.dq, 
                        self.env.q_scaled, self.env.dq_scaled,
                        self.env.u_scaled)
            
            def tm_func(t, q):
                if t>params.start_after:
                    self.tm.step(q[0], q[1]) #x,y,dx,dy
                    traj = self.tm.q_des
                    dtraj = self.tm.dq_des
                else:
                    traj = q[0]   
                    dtraj = q[1]
                return (traj, dtraj, self.tm.qf)
            
            def ctrl_func(t,x):
                q, dq, q_des, dq_des = x
                u_new = self.unode.get_control(q, dq, q_des, dq_des)*(t>=params.start_after)
                return u_new
            
            self.pend = nengo.Node(env_step, size_in=1)
            self.tm_node = nengo.Node(tm_func, size_in=2) #target, no_obs, trial#
            self.u_tot = nengo.Node(ctrl_func, size_in=4)
            nengo.Connection(self.pend[:2], self.tm_node[:2])
            
            if GTC:
                nengo.Connection(self.pend[:2], self.u_tot[:2])
                nengo.Connection(self.tm_node[:2], self.u_tot[2:4])

            else: 
                nengo.Connection(self.tm_node[:2], self.u_tot[2:])

                # the estimate is fed outside 
            
            nengo.Connection(self.u_tot, self.pend, synapse=None)
            
            self.u_scale = nengo.Node(size_in=1)
            nengo.Connection(self.pend[4], self.u_scale, synapse=None)



class EstNet(nengo.Network):
    def __init__(self, p, Xw=[], no_obs=True, label=None, **kwargs):
        super(EstNet, self).__init__(label=label)
        with self:
            def noobs(t,x):
                interval = 0.5 #secs
                gated = (t>(p.no_obs_from-interval))
                out = gated*np.clip((t-p.no_obs_from)/(2*interval), 0, 1)
                if t > p.no_obs_from +2:
                    #gradually decrease to 0
                    out = np.clip(1 - (t - p.no_obs_from - 5)/(2*interval), 0, 1)
                    
                return out

            def err_fun(t,x):
                return x[0:p.Zdim]*(1 - x[p.Zdim])
        
            def ctxtfn(t,x):
                return x[0:p.Zdim]*(1 - x[2*p.Zdim]) + x[p.Zdim:2*p.Zdim]*x[2*p.Zdim]
            
            taup = 0.1
            self.Z = nengo.Node(size_in=p.Zdim)
            self.X = nengo.Node(size_in=p.Zdim)
            self.context = nengo.Node(ctxtfn, size_in=2*p.Zdim+1)
            

            nengo.Connection(self.Z, self.context[0:p.Zdim])
            nengo.Connection(self.X, self.context[p.Zdim:2*p.Zdim])
            if no_obs:
                self.no_obs = nengo.Node(noobs, size_in=1)
                nengo.Connection(self.no_obs, self.context[2*p.Zdim])
            else:
                self.no_obs = nengo.Node(None, size_in=1)
                nengo.Connection(self.no_obs, self.context[2*p.Zdim])
            
            A, B = make_lmu(q=p.qs, theta=p.theta_s); print('Theta_s: '+str(p.theta_s) + ' Qs: '+str(p.qs))
            As = taup * A + np.eye(A.shape[0])
            Bs = taup * B
            Cs = get_weights_for_delays(0, q=p.qs)


            Q = [];    self.conn = []
            self.err = nengo.Node(err_fun, size_in=p.Zdim+1); nengo.Connection(self.no_obs, self.err[p.Zdim])
            self.adapt = nengo.Ensemble(p.n_neurons, p.Zdim*p.qs + p.Udim*p.qu)
            
            for i in range(p.Zdim):
                Q.append(nengo.Node(None, size_in=p.qs))
                
                nengo.Connection(self.context[i], Q[i], transform=Bs, synapse=taup)
                nengo.Connection(Q[i], Q[i], transform=As, synapse=taup) 
                nengo.Connection(Q[i], self.adapt[i*p.qs:(i+1)*p.qs])
                
                self.conn.append(nengo.Connection(self.adapt.neurons, self.X[i], transform=Xw[i]))
                self.conn[i].learning_rule_type = nengo.PES(learning_rate=p.Lr)                        
                nengo.Connection(self.err[i], self.conn[i].learning_rule)
                
                if p.recurr == True:
                    nengo.Connection(self.X[i], self.X[i], synapse=taup)
                
                nengo.Connection(self.Z[i], self.err[i], transform=-1, synapse=None)
                nengo.Connection(self.X[i], self.err[i], synapse=None)
                
                             
            if p.Udim !=0:
                
                self.U = nengo.Node(size_in=p.Udim)
                A, B = make_lmu(q=p.qu, theta=p.theta_u)
                Cu = get_weights_for_delays(0, q=p.qu)
                Au = taup * A + np.eye(A.shape[0])
                Bu = taup * B

                temp = p.Zdim*p.qs
                for j in range(p.Udim):
                    print("Ctrl, ",j)
                    Q.append(nengo.Node(None, size_in=p.qu))
                    nengo.Connection(Q[p.Zdim+j], Q[p.Zdim+j], transform=Au, synapse=taup)
                    nengo.Connection(self.U[j], Q[p.Zdim+j], transform=Bu, synapse=taup)
                    nengo.Connection(Q[p.Zdim+j], self.adapt[temp+j*p.qu:temp+(j+1)*p.qu])

            if p.debug == True:
                self.Check = nengo.Node(size_in=1)
                # nengo.Connection(self.adapt[4*p.qs:4*p.qs+p.qu], self.Check, transform=Cu, synapse=None)
                
                
                # self.Check2 = nengo.Node(size_in=1)
                nengo.Connection(self.adapt[1*p.qs:2*p.qs], self.Check, transform=Cs)
            
                


def Nengo_sim(params, tf):
    model = nengo.Network()
    GTC = params.GTC
    params.tf = tf

    with model:
        env = PendNet(params, GTC)            

        est = EstNet(params, Xw=params.Xw, no_obs=True)
        nengo.Connection(env.pend[2:4], est.Z, synapse=None)
        nengo.Connection(env.u_scale, est.U, synapse=None)

        if GTC:
            print("Ground truth control")

        else:
            print("Estimated control")
            
            def scale_back(t, x):
                q, dq = x
                return (q*params.q_max, dq*params.dq_max)

            nengo.Connection(est.X,  env.u_tot[:2], transform=[[params.max_q, 0], [0, params.max_dq]])

        no_obs_probe = nengo.Probe(est.no_obs, synapse=None)

        # Probes
        p_syn = 0.01
        Qp = nengo.Probe(env.pend, synapse=p_syn)
        Up = nengo.Probe(env.u_scale, synapse=p_syn)
        Minjp = nengo.Probe(env.tm_node, synapse=None)
        if params.debug:
            Cp = nengo.Probe(est.Check, synapse=p_syn)

        # # From Estimate
        Zp = nengo.Probe(est.Z, synapse=p_syn)
        Xestp = nengo.Probe(est.X, synapse=p_syn)
        Ep = nengo.Probe(est.err, synapse=p_syn)

    with nengo.Simulator(model, dt=params.dt) as sim:
        sim.run(tf)

    time = sim.trange()
    Q = sim.data[Qp]
    # X = sim.data[Xp]
    U = sim.data[Up]
    Minj = sim.data[Minjp]
    Targ = Minj[:,2]
    Z = sim.data[Zp]
    Xest = sim.data[Xestp]
    E = sim.data[Ep]
    No = sim.data[no_obs_probe]
    if params.debug:
        C = sim.data[Cp]
    else:
        C = None
    return time, Q, U, Targ, Minj, Z, Xest, E, C, No
    



def plot_states(params, probes, ind0, indf, scale = True):
    time, Q, U, Targ, Minj = probes
    Q_s = np.zeros_like(Q)
    Targ_s = np.zeros_like(Targ)
    Minj_s = np.zeros_like(Minj)
    U_s = np.zeros_like(U)


    if scale:
        print('Scaling')
        Q_s[:,0] = Q[:, 0]/params.max_q
        Q_s[:,1] = Q[:, 1]/params.max_dq
        Targ_s[:,0] = Targ[:, 0]/params.max_q
        Minj_s[:,0] = Minj[:, 0]/params.max_q
        Minj_s[:,1] = Minj[:, 1]/params.max_dq
        U_s = U/params.max_u
    else:
        Q_s = Q
        Targ_s = Targ
        Minj_s = Minj
        U_s = U

    ind0 = int(ind0/params.dt)
    if indf != -1:
        indf = int(indf/params.dt)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(time[ind0:indf], Minj_s[ind0:indf, 0], 'k--', alpha= 0.5, label='$min_jerk$')
    axs[0].plot(time[ind0:indf], Targ_s[ind0:indf],    'k', alpha=0.5,  label='$q_f$')
    axs[0].plot(time[ind0:indf], Q_s[ind0:indf, 0],    'k'  , label='$q$')
    axs[0].set_ylabel('$q$')

    axs[1].plot(time[ind0:indf], Q_s[ind0:indf, 1],     'k',  label='dq')
    # axs[1].plot(time[ind0:indf], Minj_s[ind0:indf, 1], 'r--', label='dq_des')
    axs[1].set_ylabel('$\dot{q}$')

    axs[2].plot(time[ind0:indf], U_s[ind0:indf], 'k', label='u')
    axs[1].set_xlabel('Time (s)')


def plot_states_nengo(params, probes, ind0, indf, scale = True, plot_cum_err = False):
    time, Q, U, Targ, Minj, Z, Xest, E, C, NO = probes
    Q_s = np.zeros_like(Q)
    Targ_s = np.zeros_like(Targ)
    Minj_s = np.zeros_like(Minj)
    U_s = np.zeros_like(U)

    if scale:
        print('Scaling')
        Q_s[:,0] = Z[:, 0]
        Q_s[:,1] = Z[:, 1]
        Targ_s = Targ/params.max_q
        Minj_s[:,0] = Minj[:, 0]/params.max_q
        Minj_s[:,1] = Minj[:, 1]/params.max_dq
        U_s = U
    else:
        Q_s = Q
        Targ_s = Targ
        Minj_s = Minj
        U_s = U

    ind0 = int(ind0/params.dt)
    if indf != -1:
        indf = int(indf/params.dt)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axs[0].plot(time[ind0:indf], Minj_s[ind0:indf, 0], 'k--', alpha= 0.5, label='$min-jerk$')
    axs[0].plot(time[ind0:indf], Targ_s[ind0:indf],    'k', alpha=0.5,  label='$q_f$')
    axs[0].plot(time[ind0:indf], Q_s[ind0:indf, 0],    'k'  , label='$GT$')
    axs[0].plot(time[ind0:indf], Xest[ind0:indf, 0],    'g'  , label='$q_{est}$')
    axs[0].plot(time[ind0:indf], NO[ind0:indf],    'r'  , label='$testing$')
    axs[0].set_ylabel('$q$', fontsize=16)
    axs[0].tick_params(axis='both', which='major', labelsize=16)
    
    axs[1].plot(time[ind0:indf], Q_s[ind0:indf, 1],     'k',  label='$GT$')
    # axs[1].plot(time[ind0:indf], Minj_s[ind0:indf, 1], 'r--', label='dq_des')
    axs[1].plot(time[ind0:indf], Xest[ind0:indf, 1],    'b'  , label=r'$\dot q_{est}$')
    axs[1].plot(time[ind0:indf], NO[ind0:indf],    'r')
    axs[1].set_ylabel('$\dot{q}$', fontsize=16)
    axs[1].tick_params(axis='both', which='major', labelsize=16)

    axs[2].plot(time[ind0:indf], U_s[ind0:indf], 'k', label='u')
    axs[2].set_xlabel('Time (s)', fontsize=16)
    axs[2].set_ylabel('$u$', fontsize=16)
    axs[2].set_ylim([-1.1, 1.1])
    axs[2].tick_params(axis='both', which='major', labelsize=16)
    axs[2].legend(fontsize=16)

    # Get legends from first plot and the second plot, and combine them (both pos and vel)
    handles, labels = axs[0].get_legend_handles_labels()
    handles2, labels2 = axs[1].get_legend_handles_labels()
    handles.extend(handles2)
    labels.extend(labels2)
    # Remove duplicates
    handles, labels = zip(*OrderedDict(zip(handles, labels)).items())

    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=16, bbox_to_anchor=(0.5, 1.01), fancybox=True, shadow=True)
    
    


    # if params.debug:
    #     axs[1].plot(time[ind0:indf], C[ind0:indf], 'r')
    
     # Calculate Metrics 
    ind_O = np.where(time>params.no_obs_from)[0][0]
    print('ind_O: ', ind_O)
    Cum_err_pos = np.zeros(len(time))
    Cum_err_vel = np.zeros(len(time))
    Cum_err_pos[ind_O:] = np.cumsum(np.abs(Xest[ind_O:,0] - Z[ind_O:,0]))*1e-3
    Cum_err_vel[ind_O:] = np.cumsum(np.abs(Xest[ind_O:,1] - Z[ind_O:,1]))*1e-3

    # Find when cumerr first hits 0.1 after no_obs_from
    cerr_hit_pos = np.where(Cum_err_pos[ind_O:] > 0.1)[0][0]
    cerr_hit_vel = np.where(Cum_err_vel[ind_O:] > 0.1)[0][0]
    print('cerr_hit_pos: ', cerr_hit_pos)
    print('cerr_hit_vel: ', cerr_hit_vel)
    #Plot Metrics
    if plot_cum_err:
        axs[0].plot(time[ind0:indf], Cum_err_pos[ind0:indf], 'g', label='Cum_err_pos')
        axs[1].plot(time[ind0:indf], Cum_err_vel[ind0:indf], 'g', label='Cum_err_vel')

        axs[0].plot(time[ind_O+cerr_hit_pos], Cum_err_pos[ind_O+cerr_hit_pos], 'ro', label='Cum_err_pos')
        axs[1].plot(time[ind_O+cerr_hit_vel], Cum_err_vel[ind_O+cerr_hit_vel], 'ro', label='Cum_err_vel')

        
    if scale:
        axs[0].set_xlim([time[ind0], time[indf]])
        axs[0].set_ylim([-1.1, 1.1])
        axs[1].set_xlim([time[ind0], time[indf]])
        axs[1].set_ylim([-1.1, 1.1])
        axs[2].set_ylim([-1.1, 1.1])

    # Plot error
    ind0 = 0
    indf = -1
    figE, axE = plt.subplots(2,1, figsize=(10, 10), sharex=True)
    axE[0].plot(time[ind0:indf], E[ind0:indf, 0], 'g', label='e')
    axE[0].set_ylabel('$err_{q}$', fontsize=16)
    axE[0].set_title('Angular Position Error', fontsize=16)
    axE[0].tick_params(axis='both', which='major', labelsize=16)
    axE[0].set_xlim([time[ind0], time[indf]])
    axE[0].set_ylim([-0.35, 0.35])

    axE[1].plot(time[ind0:indf], E[ind0:indf, 1], 'b', label='de')
    axE[1].set_xlabel('Time (s)', fontsize=16)
    axE[1].set_ylabel('$err_{dq}$', fontsize=16)
    axE[1].set_title('Angular Velocity Error', fontsize=16)
    axE[1].tick_params(axis='both', which='major', labelsize=16)
    axE[1].set_xlim([time[ind0], time[indf]])
    axE[1].set_ylim([-0.35, 0.35])



