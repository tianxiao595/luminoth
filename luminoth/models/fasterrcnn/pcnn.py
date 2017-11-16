import numpy as np
import sonnet as snt
import tensorflow as tf

from luminoth.models.fasterrcnn.roi_pool import ROIPoolingLayer
from luminoth.utils.losses import smooth_l1_loss
from luminoth.utils.bbox_overlap import bbox_overlap
from luminoth.utils.vars import (
    get_initializer, variable_summaries, get_activation_function
)
from luminoth.utils.bbox_transform import get_bbox_properties


class PCNN(snt.AbstractModule):
    def __init__(self, config, debug=False, seed=None, name='pcnn'):
        super(PCNN, self).__init__(name=name)
        self._layer_sizes = [
            2048, 512
        ]
        self._dropout_keep_prob = 0.5
        self._activation = get_activation_function('relu')
        self.initializer = get_initializer(config.initializer, seed=seed)
        self.regularizer = tf.contrib.layers.l2_regularizer(
            scale=config.l2_regularization_scale)

        # Debug mode makes the module return more detailed Tensors which can be
        # useful for debugging.
        self._debug = debug
        self._config = config
        self._seed = seed

    def _find_megaproposals(self, objects):
        if len(objects) == 0:
            return objects

        iou = bbox_overlap(objects, objects)
        groups = []
        for x, y in zip(*np.where(np.triu(iou - np.eye(iou.shape[0])) > 0)):
            for group in groups:
                if x in group:
                    group.add(y)
                    break
                elif y in group:
                    group.add(y)
                    break
            else:
                groups.append(set([x, y]))

        for idx in range(len(objects)):
            for group in groups:
                if idx in group:
                    break
            else:
                groups.append(set([idx]))

        group_megaproposal = []
        for group in groups:
            group_bboxes = np.stack([objects[idx] for idx in group], axis=0)
            min_values = group_bboxes.min(axis=0)
            max_values = group_bboxes.max(axis=0)
            group_megaproposal.append(
                np.array([
                    min_values[0], min_values[1], max_values[2], max_values[3]
                ])
            )

        return np.array(group_megaproposal).astype(np.float32)

    def _target(self, proposals, gt_boxes):
        iou = bbox_overlap(proposals, gt_boxes)
        max_gt_overlap = iou.max(axis=1)
        gt_overlap_idx = iou.argmax(axis=1)

        proposal_labels = np.ones((proposals.shape[0], )) * -1
        proposal_labels = np.where(
            max_gt_overlap > 0, gt_overlap_idx, proposal_labels)

        p_width, p_height, p_centerx, p_centery = get_bbox_properties(
            proposals)
        proposal_centers = np.stack([p_centerx, p_centery], axis=1)
        _, _, gt_centerx, gt_centery = get_bbox_properties(gt_boxes)
        gt_boxes_centers = np.stack([gt_centerx, gt_centery], axis=1)

        proposal_centers = proposal_centers[proposal_labels >= 0]
        p_width = p_width[proposal_labels >= 0]
        p_height = p_height[proposal_labels >= 0]
        if proposal_centers.shape[0] > 0:
            gt_boxes_centers = gt_boxes_centers[
                proposal_labels[proposal_labels >= 0].astype(np.int)]
        else:
            gt_boxes_centers = proposal_centers.copy()

        targets_x = (
            (proposal_centers[:, 0] - gt_boxes_centers[:, 0]) / p_width
        )
        targets_y = (
            (proposal_centers[:, 1] - gt_boxes_centers[:, 1]) / p_height
        )

        center_targets = np.stack([targets_x, targets_y], axis=1)

        binary_labels = np.where(
            proposal_labels >= 0,
            np.ones((proposal_labels.shape[0], )),
            np.zeros((proposal_labels.shape[0], ))
        )

        binary_labels = binary_labels.astype(np.int32)
        center_targets = center_targets.astype(np.float32)
        return binary_labels, center_targets

    def _proposal(self, proposals, center_offsets):
        # Split deltas columns into flat array
        dx = center_offsets[:, 0]
        dy = center_offsets[:, 1]

        p_width, p_height, p_centerx, p_centery = get_bbox_properties(
            proposals
        )

        # We get the center of the real box as center anchor + relative width
        pred_ctr_x = dx * p_width + p_centerx
        pred_ctr_y = dy * p_height + p_centery

        centers = np.stack([pred_ctr_x, pred_ctr_y], axis=1)

        return centers.astype(np.float32)

    def _build(self, conv_feature_map, proposals, im_shape,
               gt_boxes=None, is_training=False):

        mega_proposals = tf.py_func(
            self._find_megaproposals, [proposals], tf.float32
        )
        mega_proposals.set_shape((None, 4))
        self._roi_pool = ROIPoolingLayer(self._config.roi, debug=self._debug)

        roi_prediction = self._roi_pool(
            mega_proposals, conv_feature_map, im_shape
        )

        pooled_features = roi_prediction['roi_pool']

        flatten_features = tf.contrib.layers.flatten(pooled_features)

        net = tf.identity(flatten_features)

        self._layers = [
            snt.Linear(
                layer_size,
                name='fc_{}'.format(i),
                initializers={'w': self.initializer},
                regularizers={'w': self.regularizer},
            )
            for i, layer_size in enumerate(self._layer_sizes)
        ]

        # After flattening we are left with a Tensor of shape
        # (num_proposals, pool_height * pool_width * 512).
        # The first dimension works as batch size when applied to snt.Linear.
        for i, layer in enumerate(self._layers):
            # Through FC layer.
            net = layer(net)
            # Apply activation and dropout.
            net = self._activation(net)
            net = tf.nn.dropout(net, keep_prob=self._dropout_keep_prob)

        self._point_layer = snt.Linear(
            2, name='fc_bbox',
            initializers={'w': self.initializer},
            regularizers={'w': self.regularizer}
        )

        self._classifier_layer = snt.Linear(
            2, name='fc_classifier',
            initializers={'w': self.initializer},
            regularizers={'w': self.regularizer},
        )

        point_offsets = self._point_layer(net)
        cls_score = self._classifier_layer(net)
        cls_prob = tf.nn.softmax(cls_score, dim=1)

        variable_summaries(cls_score, 'cls_score', ['rcnn'])

        proposal_labels, center_targets = tf.py_func(
            self._target,
            [mega_proposals, gt_boxes],
            [tf.int32, tf.float32],
            stateful=False,
            name='anchor_target_layer_np'
        )

        prediction_dict = {
            'target': {
                'proposal_labels': proposal_labels,
                'center_targets': center_targets,
            }
        }

        megaproposals_centers = tf.py_func(
            self._proposal,
            [mega_proposals, point_offsets],
            tf.float32,
        )

        megaproposals_centers.set_shape((None, 2))

        center_x, center_y = tf.unstack(megaproposals_centers, axis=1)
        xmin = center_x - 20.
        ymin = center_y - 20.
        xmax = center_x + 20.
        ymax = center_y + 20.

        objects = tf.stack([xmin, ymin, xmax, ymax], axis=1)

        prediction_dict['objects'] = objects
        prediction_dict['labels'] = tf.zeros(
            tf.shape(cls_prob[:, 1]), dtype=tf.int32
        )
        prediction_dict['probs'] = cls_prob[:, 1]

        prediction_dict['centers'] = megaproposals_centers
        prediction_dict['prob'] = cls_prob[:, 1]
        prediction_dict['predictions'] = {
            'point_offsets': point_offsets,
            'cls_score': cls_score
        }

        return prediction_dict

    def loss(self, prediction_dict):
        # Classification loss
        proposal_labels = prediction_dict['target']['proposal_labels']
        proposal_labels = tf.cast(tf.reshape(
                proposal_labels, [-1]), tf.int32, name='rpn_cls_target')
        center_targets = prediction_dict['target']['center_targets']

        cls_score = prediction_dict['predictions']['cls_score']
        point_offsets = prediction_dict['predictions']['point_offsets']

        cls_target = tf.one_hot(proposal_labels, depth=2)

        ce_per_anchor = tf.nn.softmax_cross_entropy_with_logits(
            labels=cls_target, logits=cls_score
        )

        prediction_dict['cross_entropy_per_anchor'] = ce_per_anchor

        positive_labels = tf.equal(proposal_labels, 1)
        point_offsets = tf.boolean_mask(point_offsets, positive_labels)

        reg_loss_per_anchor = smooth_l1_loss(
            point_offsets, center_targets
        )

        prediction_dict['reg_loss_per_anchor'] = reg_loss_per_anchor

        return {
            'rcnn_cls_loss': tf.reduce_sum(ce_per_anchor),
            'rcnn_reg_loss': tf.reduce_sum(reg_loss_per_anchor),
        }
