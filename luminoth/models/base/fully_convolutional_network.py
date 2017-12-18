import sonnet as snt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from luminoth.models.base import BaseNetwork
from luminoth.utils.checkpoint_downloader import get_checkpoint_file
from tensorflow.contrib.slim import nets

VALID_ARCHITECTURES = set([
    'vgg_16',
])


class FullyConvolutionalNetwork(BaseNetwork):
    def __init__(self, config, parent_name=None, name='fc_network',
                 **kwargs):
        super(FullyConvolutionalNetwork, self).__init__(config, name=name,
                                                        **kwargs)
        if config.get('architecture') not in VALID_ARCHITECTURES:
            raise ValueError('Invalid architecture "{}"'.format(
                config.get('architecture')
            ))

        self._architecture = config.get('architecture')
        self._config = config
        self.parent_name = parent_name

    def _build(self, inputs, is_training=True):
        # We'll plug our SSD predictors to these endpoints
        feature_endpoints = {}

        # Build the truncated base network onto which we'll build our the extra
        # SSD feature layers
        if self.vgg_type:
            _, vgg_endpoints = nets.vgg.vgg_16(inputs, is_training=is_training,
                                               spatial_squeeze=False)
            feature_endpoints['conv4'] = vgg_endpoints[self.module_name +
                                                       '/vgg_16/conv4/conv4_3']
            base_network_truncation_endpoint = vgg_endpoints[self.module_name +
                                                             '/vgg_16/conv5/conv5_3']

        # TODO add some padding to recover the features we lost due to differences in
        #      maxpool between Caffe and tf.slim

        # Add SSD extra feature layers after the truncated base network
        with tf.variable_scope('pool5'):
            net = slim.max_pool2d(base_network_truncation_endpoint, [3, 3],
                                  stride=1, scope='pool5')
        with tf.variable_scope('conv6'):
            net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')

        endpoint_name = 'conv7'
        with tf.variable_scope(endpoint_name):
            net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
        feature_endpoints[endpoint_name] = net

        endpoint_name = 'conv8'
        with tf.variable_scope(endpoint_name):
            net = slim.conv2d(net, 256, [1, 1], scope='conv8_1')
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv8_2', padding='VALID')
        feature_endpoints[endpoint_name] = net

        endpoint_name = 'conv9'
        with tf.variable_scope(endpoint_name):
            net = slim.conv2d(net, 128, [1, 1], scope='conv9_1')
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv9_2', padding='VALID')
        feature_endpoints[endpoint_name] = net

        endpoint_name = 'conv10'
        with tf.variable_scope(endpoint_name):
            net = slim.conv2d(net, 128, [1, 1], scope='conv10_1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv10_2', padding='VALID')
        feature_endpoints[endpoint_name] = net

        endpoint_name = 'conv11'
        with tf.variable_scope(endpoint_name):
            net = slim.conv2d(net, 128, [1, 1], scope='conv11_1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv11_2', padding='VALID')
        feature_endpoints[endpoint_name] = net

        return feature_endpoints

    def load_weights(self):
        """
        Creates operations to load weigths from checkpoint for each of the
        variables defined in the module. It is assumed that all variables
        of the module are included in the checkpoint but with a different
        prefix.

        Returns:
            load_op: Load weights operation or no_op.
        """
        if self.vgg_type:
            if self._config.get('weights') is None and \
               not self._config.get('download'):
                return tf.no_op(name='not_loading_network')

            if self._config.get('weights') is None:
                # Download the weights (or used cached) if is is not specified
                # in config file.
                # Weights are downloaded by default on the ~/.luminoth folder.
                self._config['weights'] = get_checkpoint_file(
                    self._architecture)

            module_variables = snt.get_variables_in_module(
                self, tf.GraphKeys.MODEL_VARIABLES
            )

            assert len(module_variables) > 0

            scope = self.module_name
            var_to_modify = [
                             self.module_name + '/conv6/weights',
                             self.module_name + '/conv6/biases',
                             self.module_name + '/conv7/weights',
                             self.module_name + '/conv7/biases'
                             ]

            load_variables = []
            variables = (
                [(v, v.op.name) for v in module_variables if v.op.name not in
                 var_to_modify]
            )

            variable_scope_len = len(self.variable_scope.name) + 1

            for var, var_name in variables:
                checkpoint_var_name = var_name[variable_scope_len:]
                var_value = tf.contrib.framework.load_variable(
                    self._config['weights'], checkpoint_var_name
                )
                load_variables.append(
                    tf.assign(var, var_value)
                )
            with tf.variable_scope(scope, reuse=True):
                module_variables = snt.get_variables_in_module(
                    self, tf.GraphKeys.MODEL_VARIABLES
                )
                variables = (
                    [(v, v.op.name) for v in module_variables]
                )
                # TODO: make this works
                # Original weigths and biases
                # fc6_weights = tf.get_variable(#scope +
                #                               'vgg_16/fc6/weights')
                # fc6_biases = tf.get_variable(#scope +
                #                              'vgg_16/fc6/biases')
                # fc7_weights = tf.get_variable(#scope +
                #                               'vgg_16/fc7/weights')
                # fc7_biases = tf.get_variable(#scope +
                #                              'vgg_16/fc7/biases')
                # # load_variables.append(fc6_weights)
                # # load_variables.append(fc6_biases)
                # # load_variables.append(fc7_weights)
                # # load_variables.append(fc7_biases)
                # # Weights and biases to surgery
                # block6_weights = tf.get_variable(scope +
                #                                  'conv6/weights')
                # block6_biases = tf.get_variable(scope +
                #                                 'conv6/biases')
                # block7_weights = tf.get_variable(scope +
                #                                  'conv7/weights')
                # block7_biases = tf.get_variable(scope +
                #                                 'conv7/biases')
                #
                # # surgery
                # load_variables.append(
                #     tf.assign(block6_weights, fc6_weights[::3, ::3, :, ::4]))
                # load_variables.append(
                #     tf.assign(block6_biases, fc6_biases[::4]))
                # load_variables.append(
                #     tf.assign(block7_weights, fc7_weights[:, :, ::4, ::4]))
                # load_variables.append(
                #     tf.assign(block7_biases, fc7_biases[::4]))

            tf.logging.info(
                'Constructing op to load {} variables from pretrained '
                'checkpoint {}'.format(
                    len(load_variables), self._config['weights']
                ))

            load_op = tf.group(*load_variables)
            return load_op

    def get_trainable_vars(self):
        """Get trainable vars for the network.

        TODO: Make this configurable.

        Returns:
            trainable_variables: A list of variables.
        """
        return snt.get_variables_in_module(self)
