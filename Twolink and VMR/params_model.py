import numpy as np

class Get_params():
    def __init__(self):
        self.dt = 1e-3
        self.m1 = 0.1
        self.m2 = 0.1
        self.l1 = 1.0
        self.l2 = 1.0

        self.seed = 2

        self.low_q1 = -np.pi/2; self.up_q1 = 3*np.pi/4
        self.low_q2 = 0.01; self.up_q2 = np.pi-0.01

        self.low_dq1 = -1.25; self.up_dq1 = 1.25
        self.low_dq2 = -1.25; self.up_dq2 = 1.25

        self.max_torque = 0.75; self.max_speed = 3; self.limit = True
        self.g = 0

        self.kp = 200 #250
        self.kd = 20 #20

        self.u_max = np.array([0.75316371, 0.33512545])
        self.u_min = np.array([-0.69391332, -0.33126733])

        self.offset = np.array([0, 1])
        self.targ_rad = 0.5

        ### Other params for the target and vel threshold
        self.pthresh = 0.01 # 0.025
        self.vthresh = 0.01 # 0.05
        self.wait_time = 1
        self.option = 'vmr'

        self.req_rt = 1 #default should be 1
        self.time_limit = 3

        self.targ_to_cart_scale = 0.5
        self.scale_vel = 2
        # self.Vtarg_to_cart_scale = 1

        ### Params not from above (duplicates removed)
        self.blindfrom = 20
        self.start_from = 2
        self.n_neurons = 2000
        self.rot_start = 1e6
        self.rot_end   = 1e6 #[100, 116, 196, 272] # [Train, Baseline, Rotation, Washout]
        self.rot = False
        self.theta = np.deg2rad(0)

        self.theta_s = 0.2
        self.qs = 3

        self.theta_u = 0.2
        self.qu = 3

        self.Lr = 1e-4
        self.recurr = True
        self.gtc = False
        self.AC = False
        self.a_neurons = 1000

        self.Zdim = 4
        self.Udim = 2
        self.no_obs_from = self.blindfrom - 3

        self.option = 'vmr'
        self.reach_slow = False
        self.u_scale = self.max_torque

        self.debug = True

    def __str__(self):
        str_to_print = "theta_s: " + str(self.theta_s) + "\n"
        str_to_print += "theta_u: " + str(self.theta_u) + "\n"
        str_to_print += "qs: " + str(self.qs) + "\n"
        str_to_print += "qu: " + str(self.qu) + "\n \n"
        str_to_print += "Lr: " + str(self.Lr) + "\n"
        str_to_print += "recurr: " + str(self.recurr) + "\n"
        str_to_print += "gtc: " + str(self.gtc) + "\n"
        return str_to_print

