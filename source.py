from __future__ import print_function
import time as t
import tensorflow as tf
import streamlit as st
st.title('Battery Health Monitoring')

print('tensorflow version : ', tf.__version__)
print('streamlit version : ', st.__version__)

mode = st.selectbox('mode',('default', 'debug mode'))

warning_status = "ignore"
if mode == 'debug mode':
    warning_status = st.selectbox('waring status', ("always", "module", "once", "default", "error"))

import warnings
warnings.filterwarnings(warning_status)
with warnings.catch_warnings():
    warnings.filterwarnings(warning_status)
    warnings.filterwarnings(warning_status, category=UserWarning)

import numpy as np
import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from matplotlib.pyplot import cm
matplotlib_style = 'fivethirtyeight'
matplotlib.rc('xtick', labelsize=20)     
matplotlib.rc('ytick', labelsize=20)
font = {'family' : 'Dejavu Sans','size'   : 20}
matplotlib.rc('font', **font)

import pandas as pd
import random
import math

def reset_session(options=None):
  if tf.executing_eagerly():
    return
  global sess
  try:
    tf.reset_default_graph()
    sess.close()
  except:
    pass
  if options is None:
    options = default_session_options()
  sess = tf.InteractiveSession(config=options)

tfe = tf.contrib.eager
use_tf_eager = False
if use_tf_eager:
  try:
    tf.enable_eager_execution()
  except:
    reset_session()
    
#import tensorflow_probability as tfp
#tfd = tfp.distributions

def default_session_options(enable_gpu_ram_resizing=True,
                            enable_xla=False):
  config = tf.ConfigProto()
  config.log_device_placement = True
  if enable_gpu_ram_resizing:
    config.gpu_options.allow_growth = True
  if enable_xla:
    config.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_1)
  return config

def session_options(enable_gpu_ram_resizing=True, enable_xla=True):
    config = tf.compat.v1.ConfigProto()
    config.log_device_placement = True
    if enable_gpu_ram_resizing:
        config.gpu_options.allow_growth = True
    if enable_xla:
        config.graph_options.optimizer_options.global_jit_level = (
            tf.compat.v1.OptimizerOptions.ON_1)
    return config

def evaluate(tensors):
    if tf.executing_eagerly():
        return tf.contrib.framework.nest.pack_sequence_as(
            tensors,
            [t.numpy() if tf.contrib.framework.is_tensor(t) else t
             for t in tf.contrib.framework.nest.flatten(tensors)])
    return sess.run(tensors)

def reset_sess(config=None):
    if config is None:
        config = session_options()
    global sess
    tf.compat.v1.reset_default_graph()
    try:
        sess.close()
    except:
        pass
    sess = tf.compat.v1.InteractiveSession(config=config)

reset_sess()


class _TFColor(object):
  """Enum of colors used in TF docs."""
  red = '#F15854'
  blue = '#5DA5DA'
  orange = '#FAA43A'
  green = '#60BD68'
  pink = '#F17CB0'
  brown = '#B2912F'
  purple = '#B276B2'
  yellow = '#DECF3F'
  gray = '#4D4D4D'
  magenta =  '#8B008B'
  def __getitem__(self, i):
    return [
        self.red,
        self.orange,
        self.green,
        self.blue,
        self.pink,
        self.brown,
        self.purple,
        self.yellow,
        self.gray,
        self.magenta
    ][i % 10]

TFColor = _TFColor()

# Inputs: current, charge and state of degradation (delta)
# Output: voltage
@st.cache
def calcVoltage(current, charge, delta):
    a = 4 - 0.25 * (1 + delta) * current
    b = 6000 - 250 * (1 + 2*delta) * current
    if(charge < b):
        calcValue = a - 0.0005*charge + np.log(b - charge)
    else:
        calcValue = 0
    return calcValue

# Generate Discharge Profile with variable time steps
# Variable time step to respect a voltage drop (since we have very little points in the fall off region)
@st.cache
def calcTrace_constAmpCycle_varTS(current, delta, dTmax = 1.0, dVmax = 0.01):
    tList = []
    AsList = []
    AList = []
    VList = []
    cur_t = 0
    cur_As = 0
    dV_dT = dVmax/dTmax
    cur_V = calcVoltage(current, 0, delta)
    itercount = 0
    while(cur_V > 0):
        tList.append(cur_t)
        AsList.append(cur_As)
        VList.append(cur_V)
        AList.append(current)
        
        # Calculate Time step for next jump
        dT_step = dTmax
        dT_step_from_dV = dVmax/dV_dT
        if(dT_step_from_dV < dT_step):
            dT_step = dT_step_from_dV

        cur_t += dT_step
        cur_As += current*dT_step
        prev_V = cur_V
        cur_V = calcVoltage(current, cur_As, delta)
        dV_dT = math.fabs(cur_V - prev_V) * 1.0/dT_step
        #jitterPercent = 1 + random.uniform(-jitterMax, jitterMax)
        itercount += 1
    
    # Assemble into a dict
    retDict = {}
    retDict['tList'] = tList
    retDict['AsList'] = AsList
    retDict['AList'] = AList
    retDict['VList'] = VList
    return retDict

# Generate cell voltage profiles for various current discharge
st.header('Generate cell voltage profiles for various current discharge')
dTmax = st.number_input('dTmax', value=3.0)
dVmax = st.number_input('dVmax', value=0.01)
numTraces = st.number_input('numTraces', value=10)
ampDraw = np.linspace(2.5, 3.5, numTraces)
st.write('ampDraw', ampDraw)

color = cm.rainbow(np.linspace(0, 1, numTraces))

st.header('Pristine Battery Response')

curveDictList = []
for iStep in np.arange(numTraces):
    curCurveDict = calcTrace_constAmpCycle_varTS(ampDraw[iStep], 0, dTmax=dTmax, dVmax=dVmax)
    curveDictList.append(curCurveDict)

fig, axs = plt.subplots(figsize=(16, 10))
legendStrList = []
for iLoop, iCD in enumerate(curveDictList):
    if (iLoop%2 == 0):   
        axs.plot(iCD['tList'], iCD['VList'], 'b', alpha = 0.3, c=color[iLoop])
        legendStrList.append('Current = ' + '%.2f'%(ampDraw[iLoop]) + ' A' )
    
axs.set_xlabel('Time (sec)')
axs.set_ylabel('Voltage (V)')
plt.legend(legendStrList)
plt.title('Cell Potential Vs Time')
# plt.savefig('PristineBatteryDOE_V_s.png')


st.pyplot(fig)

fig, axs = plt.subplots(figsize=(16, 10))
for iLoop, iCD in enumerate(curveDictList):
    if iLoop%2 == 0:
        axs.plot(iCD['AsList'], iCD['VList'], 'b', alpha = 0.3, c=color[iLoop])
axs.set_xlabel('Capacity (AmpSec)')
axs.set_ylabel('Voltage (V)')
plt.legend(legendStrList)
plt.title('Cell Potential Vs Capacity')
# plt.savefig('PristineBatteryDOE_V_As.png')

st.pyplot(fig)
st.header('Battery Degradation')

delta_vec = [0, 0.1, 0.25, 0.55, 0.9]
cycles = [1, 100, 500, 1000, 1500]
amp = 2.5
numTraces = len(delta_vec)
color=cm.rainbow(np.linspace(0, 1, numTraces))

newcurveDictList = []
for iStep in np.arange(numTraces):
    newcurCurveDict = calcTrace_constAmpCycle_varTS(amp, delta_vec[iStep], dTmax=dTmax, dVmax=dVmax)
    newcurveDictList.append(newcurCurveDict)

fig, axs = plt.subplots(figsize=(16, 10))
legendStrList = []
for iLoop, iCD in enumerate(newcurveDictList):
    axs.plot(iCD['tList'], iCD['VList'], 'b', alpha = 0.3, c=color[iLoop])
    legendStrList.append('Cycle = ' + str(cycles[iLoop]) )
    
axs.set_xlabel('Time (sec)')
axs.set_ylabel('Voltage (V)')
plt.legend(legendStrList)
plt.title('Cell Potential Vs Time - Effect of Degradation')
# plt.savefig('Degraded_Battery_Cell_Potential.png')

st.pyplot(fig)

st.header('Building the initial model')
from sklearn.model_selection import train_test_split

# Assemble the X and Y Vectors
AsListofList = []
AListofList = []
VListofList = []
for curDict in curveDictList:
    AsListofList.append(curDict['AsList'])
    AListofList.append(curDict['AList'])
    VListofList.append(curDict['VList'])

As1DArray = np.hstack(AsListofList)
A1DArray = np.hstack(AListofList)
V1DArray = np.hstack(VListofList)

nOverall = len(As1DArray)
xOverall = np.zeros([nOverall, 2])
xOverall[:,0] = A1DArray
xOverall[:,1] = As1DArray

yOverall = V1DArray

col1, col2 = st.columns(2)
with col1:
    st.write('Xdata : ', xOverall[:10])
with col2:
    st.write('ydata : ', yOverall[:10])

st.header('Split into train and test vectors')
# Split into train and test vectors
test_size = st.number_input('test size', value=0.33) 
random_state = st.number_input('random state', value=42)
xTrain, xTest, yTrain, yTest = train_test_split(xOverall, yOverall, 
                                                  test_size=test_size, 
                                                  random_state=random_state)

st.markdown('##### train data length')
st.write(len(xTrain))
st.markdown('##### test data length')
st.write(len(xTest))

yTrain = yTrain.reshape(-1, 1)
yTest = yTest.reshape(-1, 1)

from sklearn.preprocessing import MinMaxScaler
# from sklearn.externals import joblib
import joblib

scalerX = MinMaxScaler().fit(xTrain)
scalerY = MinMaxScaler().fit(yTrain)

# Dump the scaler to the pickle file
scaler_filename = 'scaler.save'
joblib.dump([scalerX, scalerY], scaler_filename)

# Scale Train data
xTrain_scaled = scalerX.transform(xTrain)
yTrain_scaled = scalerY.transform(yTrain)
# Scale Test data
xTest_scaled = scalerX.transform(xTest)
yTest_scaled = scalerY.transform(yTest)

# print('xTrain:\t{}'.format(xTrain.shape))
# print('yTrain:\t{}'.format(yTrain.shape))
# print('xTest:\t{}'.format(xTest.shape))
# print('yTest:\t{}'.format(yTest.shape))

st.header('weight and bias wrappers')
# weight and bias wrappers
def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    with tf.compat.v1.variable_scope('', reuse=tf.compat.v1.AUTO_REUSE):
        return tf.compat.v1.get_variable('W_' + name,
                               dtype=tf.float32,
                               shape=shape,
                               initializer=initer)

def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    with tf.compat.v1.variable_scope('', reuse=tf.compat.v1.AUTO_REUSE):
        return tf.compat.v1.get_variable('b_' + name,
                               dtype=tf.float32,
                               initializer=initial)

# Create fully connected layer
def fc_layer(x, num_units, name, use_relu=True):
    """
    Create a fully-connected layer
    :param x: input from previous layer
    :param num_units: number of hidden units in the fully-connected layer
    :param name: layer name
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    in_dim = x.get_shape()[1]
    W = weight_variable(name, shape=[in_dim, num_units])
    b = bias_variable(name, [num_units])
    layer = tf.matmul(x, W)
    layer += b
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

st.write(weight_variable, bias_variable, fc_layer)

st.header('Create a Tensorflow Model')

tf.compat.v1.reset_default_graph()

# Placeholders for inputs (x) and outputs(y)
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2], name='X')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='Y')

# Create a fully-connected layer with h1 nodes as hidden layer
fc1 = fc_layer(x, 4, 'FC1', use_relu=True)
keep_prob =  tf.compat.v1.placeholder(tf.float32, name='keep_prob')
do1 = tf.nn.dropout(fc1, keep_prob=keep_prob)    
fc2 = fc_layer(do1, 16, 'FC2', use_relu=True)
fc3 = fc_layer(fc2, 64, 'FC3', use_relu=True)
fc4 = fc_layer(fc3, 16, 'FC4', use_relu=True)
fc5 = fc_layer(fc4, 4, 'FC5', use_relu=True)

# Create a fully-connected layer with n_classes nodes as output layer
output_pred = fc_layer(fc5, 1, 'OUT', use_relu=False)

# Create Handles for use in prediction module
tf.compat.v1.add_to_collection('output_pred', output_pred)
tf.compat.v1.add_to_collection('x', x)

# Hyper-parameters
epochs = st.number_input('Total number of training epochs', value=200)             # Total number of training epochs
batch_size = st.number_input('Training batch size', value=10)             # Training batch size
display_freq = st.number_input('Frequency of displaying the training results', value=50)                # Frequency of displaying the training results
learning_rate = st.number_input('The optimization initial learning rate', value=0.001)   # The optimization initial learning rate

# Define the loss function, optimizer, and accuracy
error = tf.reduce_mean(tf.math.squared_difference(y, output_pred))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(error)

def drive_upload():
    drive = st.checkbox('drive upload')
    if drive:
        # 구글 드라이브 업로드
        #from __future__ import print_function
        import pickle
        import os.path
        from googleapiclient.discovery import build
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request

        # 권한 인증 및 토큰 확인
        SCOPES = ['https://www.googleapis.com/auth/drive']
        creds = None

        # 이미 발급받은 Token이 있을 때
        if os.path.exists('token.pickle'):
           with open('token.pickle', 'rb') as token:
               creds = pickle.load(token)

        # 발급받은 토큰이 없거나 AccessToken이 만료되었을 때
        if not creds or not creds.valid:
           if creds and creds.expired and creds.refresh_token:
               creds.refresh(Request())
           else:
               flow = InstalledAppFlow.from_client_secrets_file('client_secret_249516912610-gc1lof7d9vlvj96sh0f9362ggsevbqac.apps.googleusercontent.com.json', SCOPES)
               creds = flow.run_local_server(port=0)
           # 현재 토큰 정보를 저장
           with open('token.pickle', 'wb') as token:
               pickle.dump(creds, token)

        # 연결 인스턴스 생성
        service = build('drive', 'v3', credentials=creds)

        #폴더에 파일 업데이트(사진 2개)
        from googleapiclient.http import MediaFileUpload
        file_id = '1EJqatYOMFARi7rDbHdVuhRaiHAtA1vsV'
        folder_id = '194Ie0Hg6CXqsQOWXzMyIbqIdIv6XyV0f'

        file_metadata = {
            'name': 'Initial_T.png',
            'parents': [folder_id]
        }
        media = MediaFileUpload('Initial_T.png',
                                mimetype='image/png',
                                resumable=True)
        # Retrieve the existing parents to remove
        # Move the file to the new folder
        file = service.files().update(fileId=file_id, media_body=media).execute()
        print("File ID :",file.get('id'))

        file_id = '1H5GjFMVyF46k6sx4xRtxPKcrX93DeuUA'
        folder_id = '194Ie0Hg6CXqsQOWXzMyIbqIdIv6XyV0f'

        file_metadata = {
            'name': 'Initial_A.png',
            'parents': [folder_id]
        }
        media = MediaFileUpload('Initial_A.png',
                                mimetype='image/png',
                                resumable=True)

        file = service.files().update(fileId=file_id, media_body=media).execute()
        print("File ID :",file.get('id'))

def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y

def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch

import PredictModel
# Create the op for initializing all variables
init = tf.compat.v1.global_variables_initializer()

# Create an interactive session (to keep the session in the other cells)
sess = tf.compat.v1.InteractiveSession()
# Initialize all variables
sess.run(init)
run_mode = st.selectbox('모델 수행 모드', ('default', 'train'))

if run_mode == 'train':
    train_start = st.checkbox('train start')
    training_progress_epochs = st.progress(0)
    for_one_time_epoch = 1 / epochs
    percent_complete_epochs = -for_one_time_epoch
    if train_start :

        # Number of training iterations in each epoch
        num_tr_iter = int(len(yTrain_scaled) / batch_size)
        for epoch in range(epochs):
            t.sleep(0.1)
            # streamlit progress bar
            percent_complete_epochs += for_one_time_epoch
            training_progress_epochs.progress(percent_complete_epochs)
            if (epoch % 50) == 0:
                print('Training epoch: {}'.format(epoch + 1))
            # Randomly shuffle the training data at the beginning of each epoch 
            xTrain_scaled, yTrain_scaled = randomize(xTrain_scaled, yTrain_scaled)
            for iteration in range(num_tr_iter):   
                start = iteration * batch_size
                end = (iteration + 1) * batch_size
                x_batch, y_batch = get_next_batch(xTrain_scaled, yTrain_scaled, start, end)

                # Run optimization op (backprop)
                feed_dict_batch = {x: x_batch, y: y_batch, keep_prob: 0.9}
                sess.run(optimizer, feed_dict=feed_dict_batch)

                if iteration % display_freq == 0:
                    # Calculate and display the batch loss and accuracy
                    error_batch = sess.run([error],
                                        feed_dict=feed_dict_batch)
                    if (epoch % 50) == 0:
                        print("iter {0:3d}:\t Error={1:.6f}".
                            format(iteration, error_batch[0]))

            # Run validation after every epoch
            feed_dict_test = {x: xTest_scaled, y: yTest_scaled, keep_prob: 1.0}
            error_valid = sess.run([error], feed_dict=feed_dict_test)
            # Print error after every 50 epochs
            if (epoch % 50) == 0:
                print('---------------------------------------------------------')
                print("Epoch: {0}, validation error: {1:.6f}".
                    format(epoch + 1, error_valid[0]))
                print('---------------------------------------------------------')
                st.write("Epoch: {0}, validation error: {1:.6f}".
                    format(epoch + 1, error_valid[0]))

        # Predict on training and test data
        feed_dict = {x: xTrain_scaled, keep_prob: 1.0}
        y_pred_train = sess.run(output_pred, feed_dict = feed_dict)
        feed_dict = {x: xTest_scaled, keep_prob: 1.0}
        y_pred_test = sess.run(output_pred, feed_dict = feed_dict)


        # Save the model
        #Create a saver object which will save all the variables
        saver = tf.compat.v1.train.Saver()
        model_name = './my_model'
        #Now, save the graph
        saver.save(sess, model_name)

        # import zipfile
        # import os
        # folder = os.getcwd()
        # with zipfile.ZipFile('D:/project/soh/20211228/zip/model.zip', 'w') as model_zip:
        #     for i in os.listdir(folder):
        #         if ('my_model' in i) or ('checkpoint' == i) :
        #             model_zip.write(i)
        #     model_zip.close()

        st.subheader('Initial Model - Training and Test Data Predictions')
        fig, axs = plt.subplots(figsize=(20, 10))
        # fig.suptitle('Initial Model - Training and Test Data Predictions', fontsize=32)
        plt.subplot(121)
        plt.plot(xTrain_scaled[:,1], yTrain_scaled, 'o', markersize=2, markeredgewidth=1, color='gray',fillstyle='none')
        plt.plot(xTrain_scaled[:,1], y_pred_train, '^', markersize=2, markeredgewidth=1, color='blue')
        plt.legend(['Actual', 'Pred'])
        plt.xlabel('Scaled Capacity')
        plt.ylabel('Scaled Voltage')
        plt.title('Training')
        plt.xlim([0.2, 1])
        plt.ylim([0.2, 1])
        plt.subplot(122)
        plt.plot(xTest_scaled[:,1], yTest_scaled, 'o', markersize=2, markeredgewidth=1, color='gray',fillstyle='none')
        plt.plot(xTest_scaled[:,1], y_pred_test, '^', markersize=2, markeredgewidth=1, color='red')
        plt.xlabel('Scaled Capacity')
        plt.ylabel('Scaled Voltage')
        plt.xlim([0.2, 1])
        plt.ylim([0.2, 1])
        plt.legend(['Actual', 'Pred'])
        plt.title('Test')
        plt.savefig('Initial_T.png')
        st.pyplot(fig)

        st.subheader('Initial Model - Actual Vs Predicted')
        fig, axs = plt.subplots(figsize=(20, 10))
        # fig.suptitle('Initial Model - Actual Vs Predicted', fontsize=32)
        plt.subplot(121)
        plt.plot(y_pred_train, yTrain_scaled, '^', markersize=4, markeredgewidth=4, color='blue')
        plt.legend(['Train Data'])
        plt.plot([0,1], [0, 1], 'k--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.xlim([0.2, 1])
        plt.ylim([0.2, 1])
        plt.subplot(122)
        plt.plot(y_pred_test, yTest_scaled, '^',  markersize=4, markeredgewidth=4, color='red')
        plt.plot([0,1], [0, 1], 'k--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.xlim([0.2, 1])
        plt.ylim([0.2, 1])
        plt.legend(['Test Data'])
        plt.savefig('Initial_A.png')
        st.pyplot(fig)
        drive_upload

if run_mode == 'default':
    test_start = st.checkbox('start')
    if test_start :
        from PIL import Image
        st.subheader('Initial Model - Training and Test Data Predictions')
        image = Image.open('Initial_T.png')
        image = image.resize((900, 400))
        st.image(image, use_column_width='never')

        st.subheader('Initial Model - Actual Vs Predicted')
        image = Image.open('Initial_A.png')
        image = image.resize((900, 400))
        st.image(image, use_column_width='never')
        drive_upload




# st.header('Tracking Battery Degradation')

# p_ampDraw = 2.0
# delta = 0.17
# p_dT = 60

# if(sess._closed):
#     sess = tf.InteractiveSession()
# model_name = './my_model.meta'
# saver = tf.compat.v1.train.import_meta_graph(model_name)
# saver.restore(sess, tf.train.latest_checkpoint('./'))
# [p_scalerX, p_scalerY] = joblib.load(scaler_filename) 

# # Initialize the class
# loc_pm = PredictModel.predictModel(sess, p_scalerX, p_scalerY)
    
# # Demonstrate Basic Prediction Class
# time_deg, voltage_deg = loc_pm.predictProfileBasic(p_ampDraw, delta, p_dT)
# max_indx = len(time_deg)
# # Demonstrate the Augmented Class
# tvars = tf.compat.v1.trainable_variables()
# tvars_vals = sess.run(tvars)

# tVarDict = {}
# for var, val in zip(tvars, tvars_vals):
#     # print(var.name, val)  # Prints the name of the variable alongside its value.
#     tVarDict[var.name] = val

# # Perform the base Run with the same values
# time_base, voltage_base = loc_pm.predictProfileAugmented(p_ampDraw, tVarDict, p_dT, time_deg[0], 2650)

# fig, axs = plt.subplots(figsize=(16, 10))
# axs.plot(time_deg, voltage_deg, color='r', linestyle='-')
# axs.plot(time_base, voltage_base, color='b', linestyle='--')
# plt.xlabel('Time (sec)')
# plt.ylabel('Voltage (V)')
# plt.legend(('Degraded Battery', 'Nominal Model'))
# plt.savefig('NominalVsActual.png')
# st.pyplot(fig)

# st.header('model update')

# import warnings
# from copy import deepcopy

# warning_status = "ignore" #@param ["ignore", "always", "module", "once", "default", "error"]
# warnings.filterwarnings(warning_status)
# with warnings.catch_warnings():
#     warnings.filterwarnings(warning_status, category=DeprecationWarning)
#     warnings.filterwarnings(warning_status, category=UserWarning)

# model_name_meta = model_name  #'./my_model.meta'
# saver = tf.train.import_meta_graph(model_name_meta)
# saver.restore(sess, tf.train.latest_checkpoint('./'))
# # Load the scaler
# [p_scalerX, p_scalerY] = joblib.load(scaler_filename)

# # Initialize the class
# loc_pm = PredictModel.predictModel(sess, p_scalerX, p_scalerY)

# # Create Degraded Battery Data
# #p_ampDraw = random.uniform(2.0, 4.0) # Current draw is between 2 - 4 amps
# p_ampDraw = 2.0
# print('Amp Draw: %f'%(p_ampDraw))

# #delta =  sess.run(tf.random.uniform((1, ), 0.1, 0.4)) # Pick between 10% & 40% deteriorated battery
# delta = 0.17
# print('Deterioration: %f'%(delta))
# p_dT = 60
# time_list, voltage_deg = loc_pm.predictProfileBasic(p_ampDraw, delta, p_dT)

# # Add measurement noise
# meas_std = 0.1
# num_data = len(voltage_deg)
# noise_tf = tfd.Normal(0, meas_std).sample(num_data)
# noise = sess.run(noise_tf)
# #noise[noise>0]=noise[noise>0]/10 #penalize positive values
# voltage_deg = voltage_deg + noise

# # Get Initial Model Predictions
# tvars = tf.trainable_variables()
# tvars_vals = sess.run(tvars)

# tVarDict = {}
# for var, val in zip(tvars, tvars_vals):
# # print(var.name, val)  # Prints the name of the variable alongside its value.
# tVarDict[var.name] = val

# voltage_base = []
# for i in range(len(time_list)):
# p_ampSec = p_ampDraw*time_list[i]
# volt_val = loc_pm.predictSingleRowAugmented(p_ampDraw, p_ampSec, tVarDict)
# voltage_base.append(volt_val[0, 0])


# # Initial States of the PF
# baseB_Out = tVarDict['b_OUT:0']
# baseW_Out = tVarDict['W_OUT:0']
# num_st = len(baseW_Out) + 1

# # Assign Initial State and covariance
# x_init = np.zeros((num_st, 1))
# x_init[0, 0] = baseB_Out

# for idxW in np.arange(len(baseW_Out)):
# x_init[idxW+1, 0] = baseW_Out[idxW, 0]
# print('Init State = ', x_init)
# cov_init = np.eye(num_st) * 1e-2

# # Set Process and Measurement Covariances
# Q = 1e-2
# R = meas_std**2

# # Total Number of Data Points
# num_data = len(time_list)

# cov_prev = cov_init
# x_prev = x_init

# tVarDict_new = deepcopy(tVarDict)
# volt_val_init =[]
# volt_val_updt = []

# x_Mat = np.matrix(np.zeros((num_st, num_data)))

# fig, axs = plt.subplots(figsize=(16, 10))
# axs.plot(time_list,voltage_deg-noise,'o')
# axs.plot(time_list,voltage_deg,'s')
# st.pyplot(fig)
