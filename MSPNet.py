from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import helper_tf_util
import time
import math
from utils.sampling import tf_sampling


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


def sampling(batch_size, npoint, pts, feature=None):
    '''
    inputs:
    npoint: scalar, number of points to sample
    pointcloud: B * N * D, input point cloud
    output:
    sub_pts: B * npoint * D, sub-sampled point cloud
    '''
    fps_idx = tf_sampling.farthest_point_sample(npoint, pts)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1)), (1, npoint,1))
    idx = tf.concat([batch_indices, tf.expand_dims(fps_idx, axis=2)], axis=2)
    idx.set_shape([batch_size, npoint, 2])
    if feature is None:
        return tf.gather_nd(pts, idx)
    else:
        return tf.gather_nd(pts, idx), tf.gather_nd(feature, idx)


class Network:
    def __init__(self, dataset, config):
        flat_inputs = dataset.flat_inputs
        self.config = config
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                self.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None
        
        self.ellipsoid_radii = tf.get_variable('ellipsoid_radii', shape=[3], 
                                        initializer=tf.constant_initializer([1.0, 1.0, 1.0]),
                                        trainable=True)

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            num_layers = self.config.num_layers
            self.inputs['xyz'] = flat_inputs[:num_layers]
            self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]
            self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]
            self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]
            self.inputs['features'] = flat_inputs[4 * num_layers]
            self.inputs['labels'] = flat_inputs[4 * num_layers + 1]
            self.inputs['input_inds'] = flat_inputs[4 * num_layers + 2]
            self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 3]
            self.inputs['batch_labels'] = flat_inputs[4 * num_layers + 4: 5 * num_layers + 4]

            self.labels = self.inputs['labels']
            self.batch_labels = self.inputs['batch_labels']
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]

            self.loss_type = 'wce'  # wce, lovas
            self.class_weights = DP.get_class_weights(dataset.name, dataset.num_per_class, self.loss_type)
            self.Log_file = open('log_train_' + dataset.name + '.txt', 'a')

        with tf.variable_scope('layers'):
            self.logits, self.sup_list = self.inference(self.inputs, self.is_training)

        #####################################################################
        # Ignore the invalid point (unlabeled) when calculating the loss #
        #####################################################################
        with tf.variable_scope('loss'):
            self.logits = tf.reshape(self.logits, [-1, config.num_classes])
            self.labels = tf.reshape(self.labels, [-1])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)

            self.loss = self.get_loss(valid_logits, valid_labels, self.class_weights)

        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results'):
            self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def inference(self, inputs, is_training):
        d_out = self.config.d_out
        feature = inputs['features'][..., :6]
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)
        feature_list = []

        # Encoder
        for i in range(self.config.num_layers):
            f_encoder_i = self.building_ESPE(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i], 'Encoder_layer_' + str(i), is_training)
            feature_list.append(f_encoder_i)
            f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i])
            feature = f_sampled_i

        # Top-Down Information Retrospection
        for i in range(1, self.config.num_layers):
            f_list = []
            for j in range(i-1, -1, -1):
                for k in range(i, j-1, -1):
                    f = self.nearest_interpolation(feature_list[k], inputs['interp_idx'][j])
                    f_list.append(f)
                F_fusion = self.building_LoRAFusion(f_list, d_out[j], 'LoRAFusion_layer_' + str(i) + str(j), is_training)
                feature_list[j] = F_fusion

        # Decoder
        feature_list[-1] = helper_tf_util.conv2d(feature_list[-1], feature_list[-1].get_shape()[-1].value, [1, 1], 'dmlp', [1, 1], 'VALID', True, is_training)
        for i in range(self.config.num_layers-1, 0, -1):
            f_interp_i = self.nearest_interpolation(feature_list[i], inputs['interp_idx'][i-1])
            f_concat = tf.concat([feature_list[i-1], f_interp_i], axis=3)
            f_decoder_i = helper_tf_util.conv2d_transpose(f_concat, feature_list[i-1].get_shape()[-1].value, [1, 1], 'Decoder_layer_' + str(i), [1, 1], 'VALID', bn=True, is_training=is_training)
            feature_list[i-1] = f_decoder_i

        # Classification_head
        f_layer_fc1 = helper_tf_util.conv2d(feature_list[0], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False, is_training, activation_fn=None)
        f_out = tf.squeeze(f_layer_fc3, [2])
        return f_out

    def train(self, dataset):
        flops = tf.profiler.profile(self.sess.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        print("Params: ", str(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        print("FLOPs: ", flops.total_float_ops)

        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        self.sess.run(dataset.train_init_op)
        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.logits,
                       self.labels,
                       self.accuracy]
                _, _, summary, l_out, probs, labels, acc = self.sess.run(ops, {self.is_training: True})
                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()
                if self.training_step % 50 == 0:
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                self.training_step += 1

            except tf.errors.OutOfRangeError:

                m_iou = self.evaluate(dataset)
                if m_iou > np.max(self.mIou_list):
                    # Save the best model
                    snapshot_directory = join(self.saving_path, 'snapshots')
                    makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                self.mIou_list.append(m_iou)
                log_out('Best m_IoU is: {:5.3f}'.format(max(self.mIou_list)), self.Log_file)

                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)
                # Update learning rate
                op = self.learning_rate.assign(
                    tf.multiply(self.learning_rate, self.config.lr_decays[self.training_epoch]))

                self.sess.run(op)
                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0

        print('finished')
        self.sess.close()

    def evaluate(self, dataset):

        # Initialise iterator with validation data
        self.sess.run(dataset.val_init_op)

        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0

        for step_id in range(self.config.val_steps):
            if step_id % 50 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                ops = (self.prob_logits, self.labels, self.accuracy)
                stacked_prob, labels, acc = self.sess.run(ops, {self.is_training: False})
                pred = np.argmax(stacked_prob, 1)
                if not self.config.ignored_label_inds:
                    pred_valid = pred
                    labels_valid = labels
                else:
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    labels_valid = labels_valid - 1
                    pred_valid = np.delete(pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)

            except tf.errors.OutOfRangeError:
                break

        iou_list = []
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)

        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
        log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)
        return mean_iou

    def get_loss(self, logits, labels, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        output_loss = tf.reduce_mean(weighted_losses)
        return output_loss

    def building_ESPE(self, feature, xyz, neigh_idx, d_out, name, is_training):
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_pc = self.ESPE_block(xyz, f_pc, neigh_idx, d_out, name + 'ESPE', is_training)
        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training, activation_fn=None)
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID', activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)

    def ESPE_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value
        neigh_idx = self.ellipsoid_neighbor_search(xyz, K=self.config.k_n)
        f_sh = self.spherical_harmonic_encoding(xyz, neigh_idx, l_max=6)
        f_sh_mapped = helper_tf_util.conv2d(f_sh, d_in, [1, 1], 'sh_align', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_sh_mapped], axis=-1)
        f_pc_agg = self.max_pooling(f_concat, d_out, name + 'max', is_training)
        return f_pc_agg
    
    def ellipsoid_neighbor_search(self, xyz, K):
        radii = tf.abs(self.ellipsoid_radii)
        xyz_scaled = xyz / radii
        diff = tf.expand_dims(xyz_scaled, 2) - tf.expand_dims(xyz_scaled, 1)
        dist_sq = tf.reduce_sum(tf.square(diff), axis=-1)
        neg_dist = -dist_sq
        _, neigh_idx = tf.nn.top_k(neg_dist, k=K, sorted=False)
        return neigh_idx
    
    def spherical_harmonic_encoding(self, points, l_max=6):
        # points → spherical coords
        x, y, z = tf.unstack(points, axis=-1)
        r = tf.sqrt(tf.maximum(x**2 + y**2 + z**2, 1e-12))
        theta = tf.acos(tf.clip_by_value(z / r, -1.0, 1.0))
        phi = tf.atan2(y, x)
        feats = []

        def legendre(l, m, x):
            pmm = tf.ones_like(x)
            if m > 0:
                sign = tf.constant(-1.0)**m
                fact = tf.sqrt(tf.cumprod(tf.cast(tf.range(1, 2*m, 2), tf.float32)))
                pmm = sign * (1 - x**2)**(m/2.0) * fact[-1]
            if l == m:
                return pmm
            pmmp1 = x * (2*m + 1) * pmm
            if l == m + 1:
                return pmmp1
            pll = tf.zeros_like(x)
            for ll in range(m + 2, l + 1):
                pll = ((2*ll - 1)*x*pmmp1 - (ll + m - 1)*pmm) / (ll - m)
                pmm, pmmp1 = pmmp1, pll
            return pll

        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                m_abs = abs(m)
                # normalization factor
                K = tf.sqrt((2*l + 1)/(4*math.pi) * math.factorial(l - m_abs)/math.factorial(l + m_abs))
                P_lm = legendre(l, m_abs, tf.cos(theta))

                if m == 0:
                    Y = K * P_lm
                elif m > 0:
                    Y = tf.sqrt(2.0) * K * P_lm * tf.cos(m * phi)
                else:  # m < 0
                    Y = tf.sqrt(2.0) * K * P_lm * tf.sin(m_abs * phi)
                feats.append(Y)
        sh_features = tf.stack(feats, axis=-1)
        return sh_features

    def building_LoRAFusion(self, feature_list, d_out, name, is_training, r=16):
        assert len(feature_list) >= 2
        F = feature_list[-1]
        inputs = feature_list[:-1]
        lora_features = []
        gate_values = []

        for idx, f in enumerate(inputs):
            with tf.variable_scope(name + '_lora_block_{}'.format(idx)):
                f_r = helper_tf_util.conv2d(
                    f, r, [1, 1], scope='lora_A', stride=[1, 1], padding='VALID',
                    bn=True, is_training=is_training)
                f_proj = helper_tf_util.conv2d(
                    f_r, d_out, [1, 1], scope='lora_B', stride=[1, 1], padding='VALID',
                    bn=True, is_training=is_training)
                lora_features.append(f_proj)

                # gate factor = Max(FC(Avg(f)))
                f_avg = tf.reduce_mean(f, axis=1, keepdims=True)
                gate = helper_tf_util.conv2d(
                    f_avg, d_out, [1, 1], scope='gate_fc', stride=[1, 1], padding='VALID',
                    bn=False, is_training=is_training)
                gate = tf.reduce_max(gate, axis=3, keepdims=True)
                gate_values.append(gate)
        gates = tf.concat(gate_values, axis=1)
        gates = tf.nn.softmax(gates, axis=1)

        fused = 0
        for i in range(len(lora_features)):
            g = gates[:, i:i+1, :, :]
            g = tf.tile(g, [1, tf.shape(lora_features[i])[1], 1, d_out])
            fused += lora_features[i] * g
        output = F + fused
        return output


    @staticmethod
    def random_sample(feature, pool_idx, rgb=False):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        if not rgb:
            feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        if rgb:
            pool_features = tf.reshape(pool_features, [batch_size, -1, d])
        return pool_features
    
    def cart2sph(x, y, z):
        """Cartesian → spherical: r, theta (0~π), phi (0~2π)."""
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(np.clip(z / np.where(r == 0, 1, r), -1.0, 1.0))
        phi = np.mod(np.arctan2(y, x), 2 * np.pi)
        return r, theta, phi

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features

    @staticmethod
    def max_pooling(feature_set, d_out, name, is_training):
        f_max = tf.reduce_max(feature_set, axis=-2, keepdims=True)
        f_max = helper_tf_util.conv2d(f_max, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return f_max
