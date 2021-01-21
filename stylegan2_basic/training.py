import os
import time
import argparse
import numpy as np
import tensorflow as tf

from utils import str_to_bool
from tf_utils import check_tf_version, allow_memory_growth, split_gpu_for_testing
from load_models import load_generator, load_discriminator
from dataset.get_tfrecords import get_dataset
from losses import d_logistic, d_logistic_r1_reg, g_logistic_non_saturating, g_logistic_ns_pathreg
from model.utils import merge_batch_images


def initiate_models(g_params, d_params):
    discriminator = load_discriminator(d_params, ckpt_dir=None)
    generator = load_generator(g_params=g_params, is_g_clone=False, ckpt_dir=None)
    g_clone = load_generator(g_params=g_params, is_g_clone=True, ckpt_dir=None)

    # set initial g_clone weights same as generator
    g_clone.set_weights(generator.get_weights())
    return discriminator, generator, g_clone


class Trainer(object):
    def __init__(self, t_params, name):
        self.cur_tf_ver = t_params['cur_tf_ver']
        self.use_tf_function = t_params['use_tf_function']
        self.use_custom_cuda = t_params['use_custom_cuda']
        self.model_base_dir = t_params['model_base_dir']
        self.global_batch_size = t_params['batch_size']
        self.n_total_image = t_params['n_total_image']
        self.max_steps = int(np.ceil(self.n_total_image / self.global_batch_size))
        self.n_samples = min(t_params['batch_size'], t_params['n_samples'])
        self.train_res = t_params['train_res']
        self.print_step = 100
        self.save_step = 100
        self.image_summary_step = 100
        self.reached_max_steps = False

        # copy network params
        self.g_params = t_params['g_params']
        self.d_params = t_params['d_params']

        # set optimizer params
        self.global_batch_scaler = 1.0 / float(self.global_batch_size)
        self.r1_gamma = 10.0
        self.g_opt = self.update_optimizer_params(t_params['g_opt'])
        self.d_opt = self.update_optimizer_params(t_params['d_opt'])
        self.pixel_lambda = self.g_opt['pixel_lambda']
        self.idt_lambda = self.d_opt['idt_lambda']
        self.pl_minibatch_shrink = 2
        self.pl_weight = float(self.pl_minibatch_shrink)
        self.pl_denorm = tf.math.rsqrt(float(self.train_res) * float(self.train_res))
        self.pl_decay = 0.01
        self.pl_mean = tf.Variable(initial_value=0.0, name='pl_mean', trainable=False,
                                   synchronization=tf.VariableSynchronization.ON_READ,
                                   aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        # create model: model and optimizer must be created under `strategy.scope`
        self.discriminator, self.generator, self.g_clone = initiate_models(self.g_params,
                                                                           self.d_params)

        # set optimizers
        self.d_optimizer = tf.keras.optimizers.Adam(self.d_opt['learning_rate'],
                                                    beta_1=self.d_opt['beta1'],
                                                    beta_2=self.d_opt['beta2'],
                                                    epsilon=self.d_opt['epsilon'])
        self.g_optimizer = tf.keras.optimizers.Adam(self.g_opt['learning_rate'],
                                                    beta_1=self.g_opt['beta1'],
                                                    beta_2=self.g_opt['beta2'],
                                                    epsilon=self.g_opt['epsilon'])

        # setup saving locations (object based savings)
        self.ckpt_dir = os.path.join(self.model_base_dir, name)
        self.ckpt = tf.train.Checkpoint(d_optimizer=self.d_optimizer,
                                        g_optimizer=self.g_optimizer,
                                        discriminator=self.discriminator,
                                        generator=self.generator,
                                        g_clone=self.g_clone,
                                        pl_mean=self.pl_mean)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=2)

        # try to restore
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print('Restored from {}'.format(self.manager.latest_checkpoint))

            # check if already trained in this resolution
            restored_step = self.g_optimizer.iterations.numpy()
            if restored_step >= self.max_steps:
                print('Already reached max steps {}/{}'.format(restored_step, self.max_steps))
                self.reached_max_steps = True
                return
        else:
            print('Not restoring from saved checkpoint')

    @staticmethod
    def update_optimizer_params(params):
        params_copy = params.copy()
        mb_ratio = params_copy['reg_interval'] / (params_copy['reg_interval'] + 1)
        params_copy['learning_rate'] = params_copy['learning_rate'] * mb_ratio
        params_copy['beta1'] = params_copy['beta1'] ** mb_ratio
        params_copy['beta2'] = params_copy['beta2'] ** mb_ratio
        return params_copy

    @tf.function
    def d_train_step(self, images, labels):
        with tf.GradientTape() as d_tape:
            # compute losses
            d_gan_loss, idt_loss = d_logistic(images, labels, self.generator, self.discriminator)

            # scale loss
            d_gan_loss = tf.reduce_sum(d_gan_loss) * self.global_batch_scaler
            idt_loss = tf.reduce_sum(idt_loss) * self.global_batch_scaler

            d_loss = d_gan_loss + idt_loss * self.idt_lambda

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        return d_loss, d_gan_loss, idt_loss

    @tf.function
    def d_train_step_reg(self, images, labels):
        with tf.GradientTape() as d_tape:
            # compute losses
            d_gan_loss, idt_loss, r1_penalty = d_logistic_r1_reg(images, labels, self.generator, self.discriminator)
            r1_penalty = r1_penalty * (0.5 * self.r1_gamma) * self.d_opt['reg_interval']

            # scale losses
            r1_penalty = tf.reduce_sum(r1_penalty) * self.global_batch_scaler
            d_gan_loss = tf.reduce_sum(d_gan_loss) * self.global_batch_scaler
            idt_loss = tf.reduce_sum(idt_loss) * self.global_batch_scaler

            # combine
            d_loss = d_gan_loss + idt_loss * self.idt_lambda + r1_penalty

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        return d_loss, d_gan_loss, idt_loss, r1_penalty

    @tf.function
    def g_train_step(self, images, labels):
        with tf.GradientTape() as g_tape:
            # compute losses
            g_gan_loss, pixel_loss = g_logistic_non_saturating(images, labels, self.generator, self.discriminator)

            # scale loss
            g_gan_loss = tf.reduce_sum(g_gan_loss) * self.global_batch_scaler
            pixel_loss = tf.reduce_sum(pixel_loss) * self.global_batch_scaler

            g_loss = g_gan_loss + pixel_loss * self.pixel_lambda

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        return g_loss, g_gan_loss, pixel_loss

    @tf.function
    def g_train_step_reg(self, images, labels):
        with tf.GradientTape() as g_tape:
            # compute losses
            g_gan_loss, pixel_loss, pl_penalty = g_logistic_ns_pathreg(images, labels, self.generator, self.discriminator,
                                                           self.pl_mean, self.pl_minibatch_shrink, self.pl_denorm, self.pl_decay)
            pl_penalty = pl_penalty * self.pl_weight * self.g_opt['reg_interval']

            # scale loss
            pl_penalty = tf.reduce_sum(pl_penalty) * self.global_batch_scaler
            g_gan_loss = tf.reduce_sum(g_gan_loss) * self.global_batch_scaler
            pixel_loss = tf.reduce_sum(pixel_loss) * self.global_batch_scaler

            # combine
            g_loss = g_gan_loss + pixel_loss * self.pixel_lambda + pl_penalty

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        return g_loss, g_gan_loss, pixel_loss, pl_penalty

    def train(self, train_ds, val_ds):
        if self.reached_max_steps:
            return

        # create iterator for validation dataset
        val_ds_iter = iter(val_ds)

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.ckpt_dir)

        # loss metrics
        metric_d_loss = tf.keras.metrics.Mean('d_loss', dtype=tf.float32)
        metric_g_loss = tf.keras.metrics.Mean('g_loss', dtype=tf.float32)
        metric_d_gan_loss = tf.keras.metrics.Mean('d_gan_loss', dtype=tf.float32)
        metric_g_gan_loss = tf.keras.metrics.Mean('g_gan_loss', dtype=tf.float32)
        metric_idt_loss = tf.keras.metrics.Mean('metric_idt_loss', dtype=tf.float32)
        metric_pixel_loss = tf.keras.metrics.Mean('metric_pixel_loss', dtype=tf.float32)
        metric_r1_penalty = tf.keras.metrics.Mean('r1_penalty', dtype=tf.float32)
        metric_pl_penalty = tf.keras.metrics.Mean('pl_penalty', dtype=tf.float32)

        
        # start training
        print('Start Training')
        print('max_steps: {}'.format(self.max_steps))
        t_start = time.time()
        
        zero = tf.constant(0.0, dtype=tf.float32)
        for real_images, labels in train_ds:
            step = self.g_optimizer.iterations.numpy()

            # d train step
            if (step + 1) % self.d_opt['reg_interval'] == 0:
                d_loss, d_gan_loss, idt_loss, r1_penalty = self.d_train_step_reg(real_images, labels)
            else:
                d_loss, d_gan_loss, idt_loss = self.d_train_step(real_images, labels)
                r1_penalty = zero

            # g train step
            if (step + 1) % self.g_opt['reg_interval'] == 0:
                g_loss, g_gan_loss, pixel_loss, pl_penalty = self.g_train_step_reg(real_images, labels)
            else:
                g_loss, g_gan_loss, pixel_loss = self.g_train_step(real_images, labels)
                pl_penalty = zero

            # update g_clone
            self.g_clone.set_as_moving_average_of(self.generator)

            # update metrics
            metric_d_loss(d_loss)
            metric_g_loss(g_loss)
            metric_d_gan_loss(d_gan_loss)
            metric_g_gan_loss(g_gan_loss)
            metric_idt_loss(idt_loss)
            metric_pixel_loss(pixel_loss)
            metric_r1_penalty(r1_penalty)
            metric_pl_penalty(pl_penalty)

            # get current step
            step = self.g_optimizer.iterations.numpy()

            # save to tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('d_loss', metric_d_loss.result(), step=step)
                tf.summary.scalar('g_loss', metric_g_loss.result(), step=step)
                tf.summary.scalar('d_gan_loss', metric_d_gan_loss.result(), step=step)
                tf.summary.scalar('g_gan_loss', metric_g_gan_loss.result(), step=step)
                tf.summary.scalar('idt_loss', metric_idt_loss.result(), step=step)
                tf.summary.scalar('pixel_loss', metric_pixel_loss.result(), step=step)
                tf.summary.scalar('r1_penalty', metric_r1_penalty.result(), step=step)
                tf.summary.scalar('pl_penalty', metric_pl_penalty.result(), step=step)

            # print every self.print_steps
            if step % self.print_step == 0:
                elapsed = time.time() - t_start
                print(f'[step {step}: elapsed: {elapsed:.2f}s] '
                      f'd_total: {np.sum(d_loss.numpy()):.3f}, '
                      f'd_gan_loss: {np.sum(d_gan_loss.numpy()):.3f}, '
                      f'r1_penalty: {np.sum(r1_penalty.numpy()):.3f}, '
                      f'idt_loss: {np.sum(idt_loss.numpy()):.3f}, '
                      f'pixel_loss: {np.sum(pixel_loss.numpy()):.3f}, '
                      f'g_total: {np.sum(g_loss.numpy()):.3f}, '
                      f'g_gan_loss: {np.sum(g_gan_loss.numpy()):.3f}, ')

                # reset timer
                t_start = time.time()

            # save every self.save_step
            if step % self.save_step == 0:
                self.manager.save(checkpoint_number=step)

            # save every self.image_summary_step
            if step % self.image_summary_step == 0:
                # add summary image
                real_images_val, labels_val = next(val_ds_iter)

                # fake img
                fake_images_train, fake_images_val = self.gen_samples(labels, labels_val)

                # convert to tensor image
                real_images_train = self.convert_per_replica_image(real_images)
                real_images_val = self.convert_per_replica_image(real_images_val)
                fake_images_train = self.convert_per_replica_image(fake_images_train)
                fake_images_val = self.convert_per_replica_image(fake_images_val)

                # set batch size
                real_images_train = real_images_train[:self.n_samples]
                real_images_val = real_images_val[:self.n_samples]
                fake_images_train = fake_images_train[:self.n_samples]
                fake_images_val = fake_images_val[:self.n_samples]

                # merge on batch dimension
                t_out = tf.concat([real_images_train, fake_images_train], axis=0)
                v_out = tf.concat([real_images_val, fake_images_val], axis=0)

                # make single image and add batch dimension for tensorboard
                t_out = merge_batch_images(t_out, self.train_res, rows=2, cols=self.n_samples)
                v_out = merge_batch_images(v_out, self.train_res, rows=2, cols=self.n_samples)
                t_out = np.expand_dims(t_out, axis=0)
                v_out = np.expand_dims(v_out, axis=0)

                with train_summary_writer.as_default():
                    tf.summary.image('train_out', t_out, step=step)
                    tf.summary.image('val_out', v_out, step=step)

            # check exit status
            if step >= self.max_steps:
                break

        # save last checkpoint
        step = self.g_optimizer.iterations.numpy()
        self.manager.save(checkpoint_number=step)
        return

    @tf.function
    def gen_samples(self, train_labels, val_labels):
        fake_images_train = self.g_clone(train_labels, truncation_psi=1.0, training=False)
        fake_images_val = self.g_clone(val_labels, truncation_psi=1.0, training=False)
        # # run networks
        # fake_images_05 = self.g_clone([test_z, test_labels], truncation_psi=0.5, training=False)
        # fake_images_07 = self.g_clone([test_z, test_labels], truncation_psi=0.7, training=False)

        # # merge on batch dimension: [n_samples, 3, out_res, 2 * out_res]
        # final_image = tf.concat([fake_images_05, fake_images_07], axis=2)
        return fake_images_train, fake_images_val

    @staticmethod
    def convert_per_replica_image(nchw_per_replica_images):
        as_tensor = nchw_per_replica_images
        as_tensor = tf.transpose(as_tensor, perm=[0, 2, 3, 1])
        as_tensor = (tf.clip_by_value(as_tensor, -1.0, 1.0) + 1.0) * 127.5
        as_tensor = tf.cast(as_tensor, tf.uint8)
        return as_tensor


def filter_resolutions_featuremaps(resolutions, featuremaps, res):
    index = resolutions.index(res)
    filtered_resolutions = resolutions[:index + 1]
    filtered_featuremaps = featuremaps[:index + 1]
    return filtered_resolutions, filtered_featuremaps


def main():
    # global program arguments parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--allow_memory_growth', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--debug_split_gpu', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_tf_function', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_custom_cuda', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--model_base_dir', default='./models', type=str)
    parser.add_argument('--tfrecord_dir', default='./data/tfrecords', type=str)
    parser.add_argument('--train_res', default=512, type=int)
    parser.add_argument('--shuffle_buffer_size', default=1000, type=int)
    parser.add_argument('--batch_size_per_replica', default=2, type=int)
    args = vars(parser.parse_args())

    # check tensorflow version
    cur_tf_ver = check_tf_version()

    # GPU environment settings
    if args['allow_memory_growth']:
        allow_memory_growth()
    if args['debug_split_gpu']:
        split_gpu_for_testing(mem_in_gb=4.5)

    # network params
    resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    featuremaps = [512, 512, 512, 512, 512, 256, 128, 64, 32]
    train_resolutions, train_featuremaps = filter_resolutions_featuremaps(resolutions, featuremaps, args['train_res'])
    g_params = {
        'w_dim': 512,
        'labels_dim': 259,
        'n_mapping': 8,
        'resolutions': train_resolutions,
        'featuremaps': train_featuremaps,
    }
    d_params = {
        'labels_dim': 259,
        'resolutions': train_resolutions,
        'featuremaps': train_featuremaps,
    }

    # prepare distribute strategy
    global_batch_size = args['batch_size_per_replica'] 

    # prepare dataset
    train_dataset = get_dataset(args['tfrecord_dir'], args['train_res'], batch_size=global_batch_size, is_train=True)
    val_dataset = get_dataset(args['tfrecord_dir'], args['train_res'], batch_size=global_batch_size, is_train=False)

    # training parameters
    training_parameters = {
        # global params
        'cur_tf_ver': cur_tf_ver,
        'use_tf_function': args['use_tf_function'],
        'use_custom_cuda': args['use_custom_cuda'],
        'model_base_dir': args['model_base_dir'],

        # network params
        'g_params': g_params,
        'd_params': d_params,

        # training params
        'g_opt': {'learning_rate': 0.002, 'beta1': 0.0, 'beta2': 0.99, 'epsilon': 1e-08, 'reg_interval': 8, 'pixel_lambda': 10.0/(args['train_res']*args['train_res'])},
        'd_opt': {'learning_rate': 0.002, 'beta1': 0.0, 'beta2': 0.99, 'epsilon': 1e-08, 'reg_interval': 16, 'idt_lambda':1e-3},
        'batch_size': global_batch_size,
        'n_total_image': 80000000,
        'n_samples': 3,
        'train_res': args['train_res'],
    }

    trainer = Trainer(training_parameters, name=f'stylegan2-{args["train_res"]}x{args["train_res"]}')
    trainer.train(train_dataset, val_dataset)
    return


if __name__ == '__main__':
    main()