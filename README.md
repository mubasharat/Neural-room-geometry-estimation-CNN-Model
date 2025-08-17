Listing A.1: PYTHON code used for scaling the targets (room dimensions)
1 def scale_to(x, x_min, x_max, t_min, t_max):
2"""
3 Scales x to lie between t_min and t_max
4"""
5 r = x_max - x_min
6 r_t = t_max - t_min
7 assert (math.isclose(0, r, abs_tol=np.finfo(float).eps) == False) 1
8 x_s = r_t * (x - x_min) / r + t_min
9 return x_s
Listing A.2: PYTHON code used for re-scaling the results back to the original targets (room
dimensions)
1 def scale_inv(x_s, x_min, x_max, t_min, t_max):
2"""
3 Inverse scaling
4"""
5 r = x_max - x_min
6 r_t = t_max - t_min
7 assert (math.isclose(0, r_t, abs_tol=np.finfo(float).eps) == False) 2
8 x = (x_s - t_min) * r / r_t + x_min
9 return x
Both functions, the scaling function and the inverse scaling function take as first argument the value to be
scaled as an input. The last two arguments are the scaling targets that is the new minima and maxima. Note
that the minimum and maximum of the input are not computed within the function as would normally
be the case. But have to be computed outside the function and are given as arguments. This is necessary
because we want all RIRs to be scaled in the same way. Also note 1 and 2 in both functions. This is
to check that there is indeed an input range. Errors can be discovered by noticing that inputs may just be
zero.
26
APPENDIX A. PYTHON SCRIPTS FOR THE CNN
Listing A.3: PYTHON code used for building the CNN model)
def get_cnn_model(input_shape):
1
2"""
Returns the CNN Model
3
Takes number of Samples
4
5"""
#print(input_shape)
6
inputs = Input(shape=input_shape, name= 'Input ')
7
#print(inputs.shape)
8
reshape1 = Reshape((input_shape,-1),name= 'Reshape_1 ')(inputs)
9
conv1d = Conv1D(filters=32, kernel_size=4, strides=4, name="1st_Conv1D")(reshape1)
10
banol1 = BatchNormalization(name= '1st_Batch_Normalisation ',momentum=0.9)(conv1d)
11
leaky1 = LeakyReLU(name= '1st_Leaky_ReLU ',alpha=0.1)(banol1)
12
conv2d = Conv1D(filters=32, kernel_size=2, strides=2, name="2nd_Conv1D")(leaky1)
13
banol2 = BatchNormalization(name= '2nd_Batch_Normalisation ',momentum=0.9)(conv2d)
14
leaky2 = LeakyReLU(name= '2nd_Leaky_ReLU ',alpha=0.1)(banol2)
15
conv3d = Conv1D(filters=128, kernel_size=8, strides=8, name="3rd_Conv1D")(leaky2)
16
banol3 = BatchNormalization(name= '3rd_Batch_Normalisation ',momentum=0.9)(conv3d)
17
leaky3 = LeakyReLU(name= '3rd_Leaky_ReLU ',alpha=0.1)(banol3)
18
conv4d = Conv1D(filters=128,kernel_size=2, strides=2, name="4th_Conv1D")(leaky3)
19
banol4 = BatchNormalization(name= '4th_Batch_Normalisation ',momentum=0.9)(conv4d)
20
leaky4 = LeakyReLU(name= '4th_Leaky_ReLU ',alpha=0.1)(banol4)
21
conv5d = Conv1D(filters=512,kernel_size=2, strides=2, name="5th_Conv1D")(leaky4)
22
banol5 = BatchNormalization(name= '5th_Batch_Normalisation ',momentum=0.9)(conv5d)
23
leaky5 = LeakyReLU(name= '5th_Leaky_ReLU ',alpha=0.1)(banol5)
24
conv6d = Conv1D(filters=512,kernel_size=4, strides=4, name="6th_Conv1D")(leaky5)
25
banol6 = BatchNormalization(name= '6th_Batch_Normalisation ',momentum=0.9)(conv6d)
26
leaky6 = LeakyReLU(name= '6th_Leaky_ReLU ',alpha=0.1)(banol6)
27
conv7d = Conv1D(filters=1024,kernel_size=4, strides=4, name="7th_Conv1D")(leaky6)
28
banol7 = BatchNormalization(name= '7th_Batch_Normalisation ',momentum=0.9)(conv7d)
29
leaky7 = LeakyReLU(name= '7th_Leaky_ReLU ',alpha=0.1)(banol7)
30
conv8d = Conv1D(filters=1024,kernel_size=1, strides=1, name="8th_Conv1D")(leaky7)
31
banol8 = BatchNormalization(name= '8th_Batch_Normalisation ',momentum=0.9)(conv8d)
32
leaky8 = LeakyReLU(name= '8th_Leaky_ReLU ',alpha=0.1)(banol8)
33
reshape2 = Reshape((1024,), input_shape=(1,1024), name="Reshape_2")(leaky8)
34
dense1= Dense(160, name="1st_Dense")(reshape2)
35
dense2= Dense(64, name="2nd_Dense")(dense1)
36
length = Dense(1, name= 'length ')(dense2)
37
width = Dense(1, name= 'width ')(dense2)
38
height = Dense(1, name= 'height ')(dense2)
39
model = Model(inputs=inputs, outputs=[length,width,height], name="RIR_Model")
40
return model
41
This is the PYTHON implementation of the CNN architecture described in [79, 78] and listed in 4.1. It is used
for all experiments in this thesis. Unlike Yu and Kleijn [79] who used Pytorch for their implementation,
KERAS and Tensorflow were used in this thesis. In order to come close to the implementation by Yu and
Kleijn [79] the KERAS default values for batch normalisation and leaky relu layers have been changed to the
Pytorch values.
Boris A. Reif 27
Appendix B
MATLAB scripts for RIR generation
The MATLAB scripts presented in this section show the important parts of how the base data sets have been
generated. In order to reproduce the base data sets or to expand them the RIR generator by Prof. Habets
needs to be installed on a compatible MATLAB version. The matlab version used here was R2019b with
gcc 8.3.0. The RIR generator can be downloaded from: https://www.audiolabs-erlangen.de/fau/professor/C
habets/software/rir-generator
Listing B.1: Main settings for all the MATLAB scripts
Main settings for the generation of the four base data sets with the RIR Generator by Prof. Habets
1 n = 4096; % Number of samples
2 c = 340; % Sound velocity (m/s)
3 fs = 8000; % Sample frequency (samples/s)
4 min_l = 6.0; % minum length of room (in metres)
5 max_l = 10.0; % maximum length of room (in metres)
6 min_w = 5.0; % minimum width of room (in metres)
7 max_w = 8.0; % maximum width of room (in metres)
8 min_h = 4.0; % minimum height of room (in metres)
9 max_h = 6.0; % maximum height of room (in metres)
10 num_of_rirs = 400000; % number of rooms
11
12 parfor id=0:num_of_rirs % number of rooms 3
13 % random rooms
14 l = (max_l-min_l).*rand(1,1)+min_l;
15 w = (max_w-min_w).*rand(1,1)+min_w;
16 h = (max_h-min_h).*rand(1,1)+min_h;
3 The generation of the RIRs runs parallel (parfor instead of for). Despite that the generation of 400000
RIRs takes almost two days. It is advisable to add the id to the name of the (wave) file that contains the
RIR. This ensures that if the process is killed one can easily start again and continue with the generation of
the wave files where one has left off.
28
APPENDIX B. MATLAB SCRIPTS FOR RIR GENERATION
Listing B.2: ALLFIXED
Main MATLAB code for the generation of the ALLFIXED base data set with the RIR generator by Prof. Habets
of International Audio Laboratories Erlangen. In this data set all the main parameters have been fixed to
specific values.
%% Source Positions FIXED
s_x = 1.0; % length
s_y = 1.0; % width
s_z = 1.0; % height
%% Receiver Positions FIXED
r_x = min_l - s_x; % length
r_y = min_w - s_y; % width
r_z = min_h - s_z; % height
%% Specify room
r = [r_x r_y r_z]; s = [s_x s_y s_z]; L = [l w h]; % Receiver position [x y z] (m)
% Source position [x y z] (m)
% Room dimensions [x y z] (m)
%% Specify the reflection coefficients
% random values between 0 and 1
c_1 = 0.5; % reflection coefficient
c_2 = 0.5; % reflection coefficient
c_3 = 0.5; % reflection coefficient
c_4 = 0.5; % reflection coefficient
c_5 = 0.5; % reflection coefficient
c_6 = 0.5; % reflection coefficient
beta = [c_1 c_2 c_3 c_4 c_5 c_6];
%% Generate RIR
rir = rir_generator(c, fs, r, s, L, beta, n);
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
Boris A. Reif 29
APPENDIX B. MATLAB SCRIPTS FOR RIR GENERATION
Listing B.3: ALLBUTREFL
Main MATLAB code for the generation of the ALLBUTREFL base data set with the RIR generator by Prof. Habets
of International Audio Laboratories Erlangen. In this data set all the main parameters are fixed except for
the reflection coeﬀicients.
1 %% Source Positions FIXED
2 s_x = 1.0; % length
3 s_y = 1.0; % width
4 s_z = 1.0; % height
5
6 %% Receiver Positions FIXED
7 r_x = min_l - s_x; % length
8 r_y = min_w - s_y; % width
9 r_z = min_h - s_z; % height
10
11 %% Specify room
12 r = [r_x r_y r_z]; % Receiver position [x y z] (m)
13 s = [s_x s_y s_z]; % Source position [x y z] (m)
14 L = [l w h]; % Room dimensions [x y z] (m)
15
16 %% Specify the reflection coefficients
17 % random values between 0 and 1
18 c_1 = (1-0).*rand(1,1)+0; % reflection coefficient
19 c_2 = (1-0).*rand(1,1)+0; % reflection coefficient
20 c_3 = (1-0).*rand(1,1)+0; % reflection coefficient
21 c_4 = (1-0).*rand(1,1)+0; % reflection coefficient
22 c_5 = (1-0).*rand(1,1)+0; % reflection coefficient
23 c_6 = (1-0).*rand(1,1)+0; % reflection coefficient
24 beta = [c_1 c_2 c_3 c_4 c_5 c_6];
25
26 %% Generate RIR
27 rir = rir_generator(c, fs, r, s, L, beta, n);
Boris A. Reif 30
APPENDIX B. MATLAB SCRIPTS FOR RIR GENERATION
Listing B.4: SEMIBLIND
Main MATLAB code for the generation of the SEMIBLIND base data set with the RIR generator by Prof. Habets
of International Audio Laboratories Erlangen. In this data set almost all parameters are chosen at random
except for the receiver position which is chosen semi randomly.
%% Source Positions 4
s_x = (l-1-0).*rand(1,1)+0; % length
s_y = (w-1-0).*rand(1,1)+0; % width
s_z = (h-1-0).*rand(1,1)+0; % height
%% Receiver Positions 5
r_x = s_x + ((1.0-0.25).*rand(1,1)+0.25);
r_y = s_y + ((1.0-0.25).*rand(1,1)+0.25);
r_z = s_z + ((1.0-0.25).*rand(1,1)+0.25);
%% Specify room
r = [r_x r_y r_z]; s = [s_x s_y s_z]; L = [l w h]; % Receiver position [x y z] (m)
% Source position [x y z] (m)
% Room dimensions [x y z] (m)
%% Specify the reflection coefficients
% random values between 0 and 1
c_1 = (1-0).*rand(1,1)+0; % reflection coefficient
c_2 = (1-0).*rand(1,1)+0; % reflection coefficient
c_3 = (1-0).*rand(1,1)+0; % reflection coefficient
c_4 = (1-0).*rand(1,1)+0; % reflection coefficient
c_5 = (1-0).*rand(1,1)+0; % reflection coefficient
c_6 = (1-0).*rand(1,1)+0; % reflection coefficient
beta = [c_1 c_2 c_3 c_4 c_5 c_6];
%% Generate RIR
rir = rir_generator(c, fs, r, s, L, beta, n);
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
The receiver position 5 is restricted to be at least half a meter and maximum one meter away from the
source. Note how the source position is computed 4 . This is to make sure that not only the source but
also the receiver always remain inside the simulated room.
