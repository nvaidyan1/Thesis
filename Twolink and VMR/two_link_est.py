import numpy as np
import nengo
import two_link_dyn as TL
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



class ArmNet(nengo.Network):
     def __init__(self, p, **kwargs):
        # qthresh, dqthresh, option, blindfrom, rot = 1000, block_from = 1000
        super().__init__(**kwargs)
        # Seed the random number generator
        np.random.seed(p.seed)

        ### Env
        self.env = TL.Two_Link(p)
        self.tm = TL.target_timely(p)           
        self.unode = TL.U_pd_minj(p)
        self.theta = p.theta
        self.R = np.array([[1, 0],[0, 1]])
        self.R_rot = np.array([[np.cos(self.theta), -np.sin(self.theta)],[np.sin(self.theta), np.cos(self.theta)]])
        
        with self:
            def env_step(t, u):
                self.env.step(u[0], u[1])
                return (self.env.q1, self.env.q2, self.env.dq1, self.env.dq2,
                        self.env.x[0],self.env.x[1],self.env.dx[0], self.env.dx[1],
                        self.env.q1_s, self.env.q2_s, self.env.dq1_s, self.env.dq2_s)
            
            def tm_func(t, x):
                if (t>=p.start_from):
                    self.tm.step(x) 
                return (self.tm.xminj[0], self.tm.xminj[1], 
                        self.tm.dxminj[0], self.tm.dxminj[1],
                        self.tm.opaque, self.tm.target_x[0],
                        self.tm.target_x[1], self.tm.rot, self.tm.reaches)
            
            def ctrl_func(t,x):
                #u_original:  tla.q1, tla.q2, tla.dq1, tla.dq2
                u_new = self.unode.ucalc_cart(x[0], x[1],
                                              x[2], x[3], x[4], x[5])
                return u_new*(t>=p.start_from)
            
            def ctrl_pd(t,x):
                #kpkd
                kp = p.kp 
                kd = p.kd 
                u_pd = kp*x[0:2] + kd*x[2:4]
                return u_pd*(t>1)
            
            
            def Rot_and_scale(t, x):
                x[0] = (x[0]-p.offset[0])/p.targ_to_cart_scale
                x[1] = (x[1]-p.offset[1])/p.targ_to_cart_scale
                x[2] = x[2]/p.scale_vel
                x[3] = x[3]/p.scale_vel
                
                if x[4]>0.9:  
                    R = self.R_rot
                else:
                    R = self.R                
                xnew = x
                    
                pt = R@np.array([[x[0]], [x[1]]])
                pt2 = R@np.array([[x[2]], [x[3]]])
                xnew = (pt[0,0], pt[1,0], pt2[0,0], pt2[1,0])
            
                return xnew
            
            def Opaque(t):
                return self.tm.opaque
            
            self.arm = nengo.Node(env_step, size_in=2)
            self.tm_node = nengo.Node(tm_func, size_in=4) #target, no_obs, rot
            
            self.u = nengo.Node(ctrl_pd, size_in=4)
            self.u_adapt = nengo.Ensemble(p.a_neurons,4)
            self.u_tot = nengo.Node(ctrl_func, size_in=6)
            self.no_obs = nengo.Node(None, size_in=1)
            self.rot = nengo.Node(None, size_in=1)
            self.reaches = nengo.Node(None, size_in=1)
            self.opaque = nengo.Node(Opaque, size_out=1)
            
            
            self.vision = nengo.Node(Rot_and_scale, size_in=5)
            self.prop = nengo.Node(None, size_in=4)
            
            pendarr = [4, 5, 6, 7]

            ###ctrl###
            nengo.Connection(self.u, self.u_tot[0:2], synapse=None) 
            nengo.Connection(self.u_tot, self.arm[0:2]) #ctrl u fed into pend env for step
            

            nengo.Connection(self.arm[pendarr], self.vision[0:4], synapse=None)
            nengo.Connection(self.vision, self.tm_node[0:4]) #env_step giving out x and dx
            nengo.Connection(self.rot, self.vision[4])
            nengo.Connection(self.arm[8:12], self.prop, synapse=None) 
            
            nengo.Connection(self.tm_node[4], self.no_obs, synapse=None)
            nengo.Connection(self.tm_node[7], self.rot, synapse=None)
            nengo.Connection(self.tm_node[8], self.reaches, synapse=None)
            
            self.u_scale = nengo.Node(None, size_in=2)
            
            def U_scale(u):
                #u = np.clip( (2*(u - u_min)/ (u_max-u_min)) -1, -1, 1)
                u = (2*(u - p.u_min)/ (p.u_max-p.u_min))    
                return u
            
            nengo.Connection(self.u_tot, self.u_scale, function=U_scale)
            


class EstNet(nengo.Network):
    def __init__(self, p, Xw=[], no_obs=True, label=None, **kwargs):
        super(EstNet, self).__init__(label=label)
        with self:
            def noobs(t,x):
                interval = 0.5 #secs
                gated = (t>(p.no_obs_from-interval))
                return gated*np.clip((t-p.no_obs_from)/(2*interval), 0, 1)

            def err_fun(t,x):
                return x[0:p.Zdim]*(1 - x[p.Zdim])
        
            def ctxtfn(t,x):
                return x[0:p.Zdim]*(1 - x[2*p.Zdim]) + x[p.Zdim:2*p.Zdim]*x[2*p.Zdim]
            
            taup = 0.1
            self.Z = nengo.Node(size_in=p.Zdim)
            self.X = nengo.Node(size_in=p.Zdim)
            # self.Z = nengo.Ensemble(p.n_neurons, p.Zdim)
            # self.X = nengo.Ensemble(p.n_neurons, p.Zdim)
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
            Cu = get_weights_for_delays(0, q=p.qu)


            Q = [];    self.conn = []
            self.err = nengo.Node(err_fun, size_in=p.Zdim+1); nengo.Connection(self.no_obs, self.err[p.Zdim])
            self.adapt = nengo.Ensemble(p.n_neurons, p.Zdim*p.qs + p.Udim*p.qu)
            
            for i in range(p.Zdim):
                Q.append(nengo.Node(None, size_in=p.qs))
                # Q.append(nengo.Ensemble(1000, p.qs)); print('Q'+str(i)+' neurons: '+str(1000))
                
                nengo.Connection(self.context[i], Q[i], transform=Bs, synapse=taup)
                nengo.Connection(Q[i], Q[i], transform=As, synapse=taup) 
                nengo.Connection(Q[i], self.adapt[i*p.qs:(i+1)*p.qs])
                
                self.conn.append(nengo.Connection(self.adapt.neurons, self.X[i], transform=Xw[i]))
                self.conn[i].learning_rule_type = nengo.PES(learning_rate=p.Lr)                        
                nengo.Connection(self.err[i], self.conn[i].learning_rule)
                
                if p.recurr == True:
                    print(f"Recurr {i}")
                    nengo.Connection(self.X[i], self.X[i], synapse=taup)
                
                nengo.Connection(self.Z[i], self.err[i], transform=-1, synapse=None)
                nengo.Connection(self.X[i], self.err[i], synapse=None)
                
                             
            if p.Udim !=0:
                
                self.U = nengo.Node(size_in=p.Udim)
                A, B = make_lmu(q=p.qu, theta=p.theta_u)
                Au = taup * A + np.eye(A.shape[0])
                Bu = taup * B

                temp = p.Zdim*p.qs
                for j in range(p.Udim):
                    print("Ctrl, ",j)
                    Q.append(nengo.Node(None, size_in=p.qu))
                    # Q.append(nengo.Ensemble(1000, p.qu)); print('Q'+str(j)+' neurons: '+str(1000))
                    nengo.Connection(Q[p.Zdim+j], Q[p.Zdim+j], transform=Au, synapse=taup)
                    nengo.Connection(self.U[j], Q[p.Zdim+j], transform=Bu, synapse=taup)
                    nengo.Connection(Q[p.Zdim+j], self.adapt[temp+j*p.qu:temp+(j+1)*p.qu])

            if p.debug == True:
                self.Check = nengo.Node(size_in=1)
                nengo.Connection(self.adapt[4*p.qs:4*p.qs+p.qu], self.Check, transform=Cu, synapse=None)
                
                
                self.Check2 = nengo.Node(size_in=1)
                nengo.Connection(self.adapt[2*p.qs:3*p.qs], self.Check2, transform=Cs, synapse=None)