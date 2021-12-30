import numpy as np
import tensorflow as tf

class predictModel(object):
    def __init__(self, sess, scalerX, scalerY,
                 pred_node_name='output_pred', input_node_name='x'):
        self.sess = sess
        self.scalerX = scalerX
        self.scalerY = scalerY
        # Get the Graph Object and extract the prediction node
        self.graph = sess.graph
        self.pred_node = tf.compat.v1.get_collection(pred_node_name)[0]
        self.x = tf.get_collection(input_node_name)[0]
        self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')

    def predictSingleRowBasic(self, p_ampDraw, p_ampSec):
        X_pred = np.array([p_ampDraw, p_ampSec])
        X_pred_reshape = X_pred.reshape(1, -1)
        s_X_pred = self.scalerX.transform(X_pred_reshape)
        s_p_output = self.sess.run(self.pred_node, feed_dict={self.x: s_X_pred, self.keep_prob: 1.0})
        # Invert the scalar to get actual prediction
        p_output = self.scalerY.inverse_transform(s_p_output)
        return p_output

    def predictSingleRowAugmented(self, p_ampDraw, p_ampSec, nameValueDict):
        X_pred = np.array([p_ampDraw, p_ampSec])
        X_pred_reshape = X_pred.reshape(1, -1)
        s_X_pred = self.scalerX.transform(X_pred_reshape)

        feed_dict = {self.x: s_X_pred, self.keep_prob: 1.0}
        for curKey in list(nameValueDict.keys()):
            curTFNode = self.graph.get_tensor_by_name(curKey)
            feed_dict[curTFNode] = list(nameValueDict[curKey])
        s_p_output = self.sess.run(self.pred_node, feed_dict=feed_dict)
        # Invert the scalar to get actual prediction
        p_output = self.scalerY.inverse_transform(s_p_output)
        return p_output

    def predictProfileBasic(self, p_ampDraw, alpha, p_dT):

        # Predict the Initial point
        t_list = []
        v_list = []
        loc_t = 0
        loc_as = 0
        runLoop = True
        while runLoop:
            p_ampDraw_1 = p_ampDraw * (1 + alpha)
            loc_as_1 = loc_as * (1 + 2 * alpha)
            X_pred = np.array([p_ampDraw_1, loc_as_1])
            X_pred_reshape = X_pred.reshape(1, -1)
            s_X_pred = self.scalerX.transform(X_pred_reshape)
            s_p_output = self.sess.run(self.pred_node, feed_dict={self.x: s_X_pred, self.keep_prob: 1.0})
            p_output = self.scalerY.inverse_transform(s_p_output)

            if (p_output[0, 0] < 6):
                runLoop = False
            else:
                t_list.append(loc_t)
                v_list.append(p_output[0, 0])

                loc_t += p_dT
                loc_as += p_ampDraw * p_dT

        return t_list, v_list

    def predictProfileAugmented(self, p_ampDraw, nameValueDict, p_dT, tStart, tEnd):
        loc_t = tStart
        loc_as = p_ampDraw * (loc_t)
        time_list = []
        voltage_list = []
        runLoop = True
        while (loc_t <= tEnd) & runLoop:
            X_pred = np.array([p_ampDraw, loc_as])
            X_pred_reshape = X_pred.reshape(1, -1)
            s_X_pred = self.scalerX.transform(X_pred_reshape)

            # [s_p_ampDraw, s_p_ampSec] = p_scalerX.transform(X_pred_reshape)
            feed_dict = {self.x: s_X_pred, self.keep_prob: 1.0}
            for curKey in list(nameValueDict.keys()):
                curTFNode = self.graph.get_tensor_by_name(curKey)
                feed_dict[curTFNode] = list(nameValueDict[curKey])

            s_p_output = self.sess.run(self.pred_node, feed_dict=feed_dict)
            # Invert the scalar to get actual prediction
            p_output = self.scalerY.inverse_transform(s_p_output)
            
            time_list.append(loc_t)
            voltage_list.append(p_output[0, 0])

            loc_t += p_dT
            loc_as += p_ampDraw * p_dT

            # if (p_output[0, 0] < 6):
            #    runLoop = False
            #else:
            #    time_list.append(loc_t)
            #    voltage_list.append(p_output[0, 0])

            #    loc_t += p_dT
            #    loc_as += p_ampDraw * p_dT

        return time_list, voltage_list
