import numpy as np
try:
    import params_model as p
except:
    print("params not imported directly")

class Two_Link(object):
    def __init__(self, p):
        
        # 1-upperarm, 2-forearm
        self.m1 = p.m1;  self.m2 = p.m2 
        self.l1 = p.l1;  self.l2 = p.l2
        self.dt = p.dt
        self.g = p.g
        #limits
        self.low_q1 = p.low_q1; self.up_q1 = p.up_q1
        self.low_q2 = p.low_q2; self.up_q2 = p.up_q2
        self.low_dq1 = p.low_dq1; self.up_dq1 = p.up_dq1
        self.low_dq2 = p.low_dq2; self.up_dq2 = p.up_dq2
        
        self.max_torque = p.max_torque
        self.max_speed = p.max_speed
        self.limit = p.limit
        self.offset = p.offset; self.targ_rad = p.targ_rad
        self.reset(p.seed)
        self.update_cart()
        self.scale()
        
    def reset(self, seed):
        self.rng = np.random.RandomState(seed=seed)
        self.q1 = 0.5236400976443022 #np.pi/6 #0*self.rng.uniform(self.low_q1, self.up_q1)
        self.q2 = 2.09436288969949 #np.pi/2 #0*self.rng.uniform(self.low_q2, self.up_q2)
        self.dq1 = 0.0 #self.rng.uniform(-1, 1)
        self.dq2 = 0.0 
    
    def update_cart(self):
        self.e =  np.array([self.l1 * np.cos(self.q1), self.l1 * np.sin(self.q1)])
        self.x =  self.e + np.array([self.l2 * np.cos(self.q1+self.q2), self.l2 * np.sin(self.q1+self.q2)])
        self.dx= np.array([-self.l1*np.sin(self.q1)*self.dq1 - self.l2*np.sin(self.q1+self.q2)*(self.dq1+self.dq2) ,
                           self.l1*np.cos(self.q1)*self.dq1 + self.l2*np.cos(self.q1+self.q2)*(self.dq1+self.dq2) ])
        
        self.x_scaled = np.clip((self.x - self.offset)/self.targ_rad, -1, 1)

    def scale(self):
        #This scaling have to be fixed if different radius of reach
        self.q1_s = (2*(self.q1 + 0.15638416)/ (1.0814649+0.52302206)) -1
        self.q2_s = (2*(self.q2 - 1.41167524)/ (2.65949297-1.41167524)) -1
        self.dq1_s = (2*(self.dq1 +1.3713473)/ (1.39191433+1.3713473)) -1
        self.dq2_s = (2*(self.dq2 +1.36787882)/ (1.355404+1.36787882)) -1
        
                    
    def step(self, u1,u2):
        # u1 = np.clip(u1,-self.max_torque, self.max_torque)
        # u2 = np.clip(u2,-self.max_torque, self.max_torque)
        U = np.matrix([[u1],[u2]])
        
        M11 = (self.m1+self.m2)*self.l1**2 + self.m2*self.l2**2 + 2*self.m2*self.l1*self.l2*np.cos(self.q2)
        M12 = self.m2*self.l2**2 + self.m2*self.l1*self.l2*np.cos(self.q2)
        M21 = M12
        M22 = self.m2*self.l2**2
        M = np.matrix([[M11,M12],[M21,M22]])
        
        C1 = -self.m2*self.l1*self.l2*(2*self.dq1*self.dq2+self.dq2**2)*np.sin(self.q2)
        C2 = self.m2*self.l1*self.l2*self.dq1**2*np.sin(self.q2)
        C = np.matrix([[C1],[C2]]);
        G1 = (self.m1+self.m2)*self.g*self.l1*np.cos(self.q1) + self.m2*self.g*self.l2*np.cos(self.q1+self.q2)
        G2 = self.m2*self.g*self.l2*np.cos(self.q1+self.q2)
        G = np.matrix([[G1],[G2]]);
        ACC = np.linalg.inv(M) * (U-C-G)
        ddq1,ddq2 = ACC[0,0],ACC[1,0]
        
        tempq1 = self.q1
        tempq2 = self.q2
        tempdq1 = self.dq1
        tempdq2 = self.dq2
        
        self.dq1 += ddq1*self.dt;  self.dq2 += ddq2*self.dt
        self.q1 += self.dq1*self.dt;  self.q2 += self.dq2*self.dt
        
        if self.limit:
            self.q1 = np.clip(self.q1, self.low_q1, self.up_q1)
            self.q2 = np.clip(self.q2, self.low_q2, self.up_q2)
            self.dq1 = np.clip(self.dq1, self.low_dq1, self.up_dq1)
            self.dq2 = np.clip(self.dq2, self.low_dq2, self.up_dq2)
        self.update_cart();
        self.scale()
        

                   
    
class U_pd_minj(object):
    def __init__(self, p):
        self.kp = p.kp #250 
        self.kd = p.kd #18

        self.l1 = p.l1
        self.l2 = p.l2
        self.m1 = p.m1
        self.m2 = p.m2
        self.max_torque = p.max_torque
    

    def ucalc_cart(self, ux, uy, q1, q2, dq1, dq2):
        # print(ux, uy, q1, q2, dq1, dq2, "This")
        """This is in task space"""
        Jee = np.array([[-self.l1*np.sin(q1)-self.l2*np.sin(q1+q2),-self.l2*np.sin(q1+q2)],
                        [self.l1*np.cos(q1)+self.l2*np.cos(q1+q2),self.l2*np.cos(q1+q2)]]) 
        
        M11 = (self.m1+self.m2)*self.l1**2 + self.m2*self.l2**2 + 2*self.m2*self.l1*self.l2*np.cos(q2)
        M12 = self.m2*self.l2**2 + self.m2*self.l1*self.l2*np.cos(q2);M21 = M12
        M22 = self.m2*self.l2**2
        M = np.matrix([[M11,M12],[M21,M22]])
        Mxee = np.linalg.pinv(Jee @  np.linalg.pinv(M) @Jee.T)

        pa=np.matrix([q1,q2])
        pw=np.matrix([dq1,dq2])
        pv =  Jee@pw.T
#         self.pv = pv
        U = Jee.T @ Mxee @(np.array([[ux],[uy]]) )#target_y-y
        U = np.clip(U, -self.max_torque, self.max_torque)
        return np.array([U[0,0], U[1,0]])


class target_timely (object):
    def __init__(self, p, max_time = 2):
        """
        Creates an environment to give targets.
        The targets are time based to check across experiments.
        """
        
        # Meta choices: Targets to pseudo random directions 
        self.seed = 42
        self.option = p.option # up, vmr, rand
        self.req_rt = 1.0
        self.wait_time = p.wait_time
        self.max_time = self.req_rt + self.wait_time
        
        # Plant specific 
        self.target_x = np.array([0,0])
        self.global_time = 0
        
        # Target specific
        self.reaches = 0 # Number of times the target has changed
        self.hits = 0    # Number of trials when the plant reached the target within max_time
        self.targ_seq = np.linspace(0, 7, 8)
        np.random.shuffle(self.targ_seq)
        self.opaque = False; 
        self.rot_start = p.rot_start; self.rot_end = p.rot_end
        self.rot = False
        self.rho = 0
        self.dt = 1e-3
        self.avg_vel = 0.5
        self.toggle_rot = False
        self.pthresh = p.pthresh
        self.vthresh = p.vthresh
        self.p = p

        self.home = False
        
        self.reach_slow = p.reach_slow
        if p.reach_slow:
            self.reach_slow_factor = 0.1
        else:
            self.reach_slow_factor = 1
            
        self.targ_rad = p.targ_rad*(self.reach_slow_factor)
        
        #Minimum jerk trajectory
        x = self.target_x
        self.xi = x
        self.rt = 0 #reach time
        self.phi=0
        self.trials = 0;  # One reach to target and back home
        self.reset(1,1)
        
        self.targ_hit = False

    def reset(self, train_seed=42, test_seed=42):
        self.rng_train = np.random.RandomState(train_seed) 
        self.rng_test = np.random.RandomState(test_seed)
        self.xminj = self.xi
        self.dxminj = self.xi*0
    
    def minjt(self, xi, xf, t, tf):
        t = np.clip(t, 0, tf)
        r = (t/tf)
        self.xminj = xi + (xf - xi) * (10*r**3 - 15*r**4 + 6*r**5)
        self.dxminj  = (1/tf)*(xf- xi) * (30*r**2 -60*r**3 + 30*r**4)  

    def step(self, X):
        #1 = training, #2 = transition to common test, #test
        x, y, dx, dy = X 
        self.global_time += self.dt
        self.rt += self.dt
        
        dist_at_targ = np.sqrt((x - self.target_x[0])**2 + (y - self.target_x[1])**2)
        vel_at_targ =  np.sqrt(dx**2 + dy**2)
        dist_from_offset = np.sqrt(x **2 + y**2)
        
        # Opaque feedback only after trials>rot
        if (dist_from_offset>(0.10*self.targ_rad) and dist_from_offset<(0.85*self.targ_rad) and self.rot and not self.home):
            self.opaque = True
        else:
            self.opaque = False

        if (self.global_time>2) and ((dist_at_targ<=self.pthresh)and(vel_at_targ<=self.vthresh)) and not self.targ_hit:
            self.targ_hit = True
        
        if (self.rt>self.max_time) or self.targ_hit:
            self.reaches += 1
            

            # Increment reachnum if didn't reach for home
            if self.home: 
                self.trials += 1
            
            if (self.trials>self.rot_start) and (self.trials<=self.rot_end):
                self.rot = True; 
            else:
                self.rot = False

            if self.targ_hit:
                self.hits += 1
                print(f"Hit")
                self.targ_hit = False
                # As we hit more targets, the target gets bigger upto the max targ_rad
                if self.reach_slow and not self.p.gtc:
                    # INcrement reach slow factor by 0.1 until 1
                    self.reach_slow_factor += 0.05
                    self.reach_slow_factor = np.clip(self.reach_slow_factor, 0, 1)
                    self.targ_rad = self.p.targ_rad*(self.reach_slow_factor); print(f"targ_rad: {self.targ_rad} and reach_slow_factor: {self.reach_slow_factor:.2f}")
                
            # Change target
            if (self.option == 'up'):
                self.rho = self.targ_rad*self.rng_train.uniform(0, 1)*(self.reaches%2) 
                self.phi = np.pi/2
                
            elif (self.option == 'rand'):
                self.rho = self.targ_rad*self.rng_train.uniform(0, 1)*(self.reaches%2)
                self.phi = self.rng_train.uniform(0, 2*np.pi) #np.pi/2#
            else:
#                 print("vmr")
                self.rho = self.targ_rad*((1+self.reaches)%2) 
                self.phi = (self.targ_seq[(self.reaches%16)//2])*np.pi/4
                self.home = self.rho == 0
                        
            
            self.xi = np.array([x, y]) # The place to move from is where the hand currently is 
            self.target_x = np.array([self.rho * np.cos(self.phi), self.rho * np.sin(self.phi)])
        
            if self.reaches%5==0:
                print(" Reach num: "+ str(self.reaches) +" Hits: %.2f" %(self.hits) + " Trials: %.2f" %(self.trials)+ " targ_rad %.2f" %(self.targ_rad)
                      + " Last rt: %.2f" %(self.rt) + " pos_err: %.2f" %(dist_at_targ) + " vel_err %.2f" %(vel_at_targ) + " global_time %.2f" %(self.global_time))
                
            self.rt = 0
        self.minjt(self.xi, self.target_x, self.rt, self.req_rt)
    