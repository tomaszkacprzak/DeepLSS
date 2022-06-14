# Copyright (C) 2020 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created on Nov 12 2020
author: Tomasz Kacprzak
"""
import os, sys, warnings, argparse, h5py, numpy as np, time, itertools, random, shutil
from collections import OrderedDict
import tensorflow as tf
from deep_lss import utils_logging, utils_io, utils_arrays
from deep_lss.configs import params as params_sims
from deep_lss.filenames import *
from deep_lss.networks import losses
from deep_lss.apps.run_recordmaker import parse_inverse as parse_inverse_cosmo 
from deep_lss.apps.run_astrosampler import parse_inverse as parse_inverse_astro

# tensorflow config
tf.config.run_functions_eagerly(False)

# tf.data.experimental.enable_debug_mode()

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('once', category=UserWarning)
LOGGER = utils_logging.get_logger(__file__)

def setup(args):

    description = 'Train DeepLSS networks'
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument('-v', '--verbosity', type=str, default='info', choices=('critical', 'error', 'warning', 'info', 'debug'), 
                        help='logging level')
    parser.add_argument('--filename_config', type=str, required=True, 
                        help='configuration yaml file')
    parser.add_argument('--dir_output', type=str, required=True, 
                        help='output dir')
    parser.add_argument('--test', action='store_true',
                        help='test mode')
    parser.add_argument('--models_filter', type=str, required=False, default=None,
                        help='only use these models')
    parser.add_argument('--continue_training', action='store_true',
                        help='continue training of the last model')
    parser.add_argument('--copy_dataset', action='store_true',
                        help='copy dataset')
    parser.add_argument('--benchmark', action='store_true',
                        help='if to benchmark using cached dataset')
    parser.add_argument('--benchmark_readout', action='store_true',
                        help='if to benchmark dataset readout and parsing')


    argk, _ = parser.parse_known_args(args)
    argk.resources = resources(args)

    utils_logging.set_all_loggers_level(argk.verbosity)

    if argk.test:
        tf.config.run_functions_eagerly(True)
        tf.config.set_soft_device_placement(False)
        tf.data.experimental.enable_debug_mode()
        LOGGER.warning('========> test, running functions eagerly')
        


    return argk


def resources(args):

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=str, default='gwen_short', choices=('gwen_short', 'gwen_short2', 'gwen_short1',  'gwen', 'gpu', 'gpu_short', 'cpu'))  
    argk, _ = parser.parse_known_args(args)

    res = dict(main_nsimult=1,
               main_memory=4000,
               main_time_per_index=1, # hours
               main_nproc=4,
               main_scratch=6500,
               main_ngpu=1)

    
    
    if argk.queue == 'gwen_short':

        res['main_time_per_index']=1 # hours
        res['main_ngpu']=4
        res['main_nproc']=16*res['main_ngpu']
        res['main_memory']=3900
        res['pass'] = {'cluster':'gmerlin6', 'account':'gwendolen', 'partition':'gwendolen', 'gpus-per-task':res['main_ngpu'], 'cpus-per-gpu':16}

    elif argk.queue == 'gwen_short2':

        res['main_time_per_index']=1 # hours
        res['main_ngpu']=2
        res['main_nproc']=16*res['main_ngpu']
        res['main_memory']=3900
        res['pass'] = {'cluster':'gmerlin6', 'account':'gwendolen', 'partition':'gwendolen', 'gpus-per-task':res['main_ngpu'], 'cpus-per-gpu':16}

    elif argk.queue == 'gwen_short1':

        res['main_time_per_index']=1 # hours
        res['main_ngpu']=1
        res['main_nproc']=16*res['main_ngpu']
        res['main_memory']=3900
        res['pass'] = {'cluster':'gmerlin6', 'account':'gwendolen', 'partition':'gwendolen', 'gpus-per-task':res['main_ngpu'], 'cpus-per-gpu':16}


    elif argk.queue == 'gwen':

        res['main_time_per_index']=8 # hours
        res['main_ngpu']=4
        res['main_nproc']=16*res['main_ngpu']
        res['main_memory']=3900
        res['pass'] = {'cluster':'gmerlin6', 'account':'gwendolen', 'partition':'gwendolen-long', 'gpus-per-task':res['main_ngpu'], 'cpus-per-gpu':16}

    elif argk.queue == 'gpu':

        res['main_time_per_index']=24 # hours
        res['main_ngpu']='geforce_rtx_2080_ti:1'
        res['main_nproc']=4
        res['pass'] = {'cluster':'gmerlin6', 'partition':'gpu', 'gpus-per-task':'geforce_rtx_2080_ti:1', 'cpus-per-gpu':4}

    elif argk.queue == 'gpu_short':

        res['main_time_per_index']=2 # hours
        res['main_ngpu']='geforce_rtx_2080_ti:1'
        res['main_nproc']=4
        res['pass'] = {'cluster':'gmerlin6', 'partition':'gpu-short', 'gpus-per-task':'geforce_rtx_2080_ti:1', 'cpus-per-gpu':4}

    elif argk.queue == 'cpu':

        res['main_time_per_index']=24 # hours
        res['main_ngpu']=0
        res['main_nproc']=24
        res['pass'] = dict(cluster='merlin6')

    return res

def main(indices, args):

    print('CUDA_VISIBLE_DEVICES',os.environ['CUDA_VISIBLE_DEVICES'])

    # some config variables
    args = setup(args)
    ctx = utils_io.load_context(args.filename_config, args.models_filter)
    utils_io.test_tensorflow()
    res = args.resources
    walltime = res['main_time_per_index']*60
    time_start = time.time()

    # make output dirs
    utils_io.robust_makedirs(args.dir_output)
    dirpath_checkpoints = get_dirpath_checkpoints(args.dir_output)
    utils_io.robust_makedirs(dirpath_checkpoints)
    dirpath_logs = get_dirpath_lssnet_logs(args.dir_output)
    utils_io.robust_makedirs(dirpath_logs)
    seq = 'nontomo' if len(ctx.load_redshift_bins)==1 else 'tomo' 
    dir_cosmo, dir_astro = utils_io.copy_dataset_to_local_scratch(args, ctx, sequence=seq, n_files=4 if args.benchmark else None)

    LOGGER.critical('=================> lssnet training - starting')
    LOGGER.timer.reset()

    netnames_all = [n for n in ctx.netnames] # copy

    for index in indices:

        LOGGER.info(f'running nets {ctx.netnames}')

        # the main bit
        run_lssnet_training(ctx=ctx, 
                            dir_output=args.dir_output, 
                            dirpath_logs=dirpath_logs,
                            dirpath_checkpoints=dirpath_checkpoints,
                            dirpath_cosmo=dir_cosmo,
                            dirpath_astro=dir_astro,
                            continue_training=args.continue_training if index == 0 else True,
                            walltime=walltime,
                            test=args.test,
                            benchmark=args.benchmark,
                            benchmark_readout=args.benchmark_readout,
                            time_cleanup=10,
                            time_start_job=time_start,
                            seed=index*1231)

    # cleanup
    if args.copy_dataset and not args.test:
        utils_io.robust_remove(dir_cosmo)
        utils_io.robust_remove(dir_astro)

    LOGGER.critical(f'=================> lssnet training - done in {LOGGER.timer.elapsed()}')


    yield 0



def run_lssnet_training(ctx,
                        dirpath_cosmo,
                        dirpath_astro,
                        dir_output,
                        dirpath_logs,
                        dirpath_checkpoints,
                        test=False,
                        benchmark=False,
                        benchmark_readout=False,
                        continue_training=False,
                        walltime=None,
                        time_cleanup=20,
                        time_start_job=0,
                        seed=42):

    from deep_lss.networks import losses
    from deep_lss.networks import multilssnet


    # get distribution strategy
    if test:
        strategy = tf.distribute.get_strategy()
        LOGGER.warning('test mode, using no distributed strategy')
    else:
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(num_packs=1))

    if benchmark:
        ctx.training['n_batches_per_epoch'] = 200
        ctx.training['n_epochs'] = 3
        LOGGER.warning('benchmark mode, using 3 epochs with 200 batches')

    with strategy.scope():

        # seed generator
        tf.random.set_global_generator(tf.random.Generator.from_seed(seed))

        nets = utils_io.get_networks(ctx=ctx,
                                     dirpath_checkpoints=get_dirpath_checkpoints(dir_output) if continue_training else None,
                                     dirpath_logs=dirpath_logs)

        multinet = multilssnet.MultiLssNET(nets=nets, 
                                           tag=ctx.tag)

        LOGGER.info(f'=============================> created multilssnet model with {multinet.n_nets} nets:')
        multinet.summary()

        # problem variable and settings
        mosaic =  ctx.training['mosaic']
        img_size = ctx.img_size
        n_channels = ctx.n_channels
        n_fields_per_survey = ctx.n_fields_per_survey
        batch_size =  ctx.training['batch_size']
        assert batch_size % strategy.num_replicas_in_sync == 0, 'batch size should be divisible by num_replicas_in_sync for distributed strategy'
        n_epochs = ctx.training['n_epochs']
        n_batches = ctx.training['n_batches_per_epoch']
        n_nets = len(nets)
        output_selects = [net.output_select for net in nets]
        batch_size_split = batch_size//strategy.num_replicas_in_sync
        valid_freq = 10 # evaluate validation batch each valid_freq tranining batches
        LOGGER.info(f'batch_size={batch_size} n_fields_per_survey={n_fields_per_survey} batch_size_split={batch_size_split}')

        if test:
            n_batches=20
            batch_size=10
            batch_size_split=10
            n_epochs=4
            LOGGER.warning('=================> test!')

        # get training data files
        files_astro = [os.path.join(dirpath_astro, f) for f in np.sort(os.listdir(dirpath_astro))]
        files_cosmo = [os.path.join(dirpath_cosmo, f) for f in np.sort(os.listdir(dirpath_cosmo))]

        # train test split, use 5 files for validation 
        files_cosmo_valid = files_cosmo[:5]
        files_cosmo_train = files_cosmo[5:]  
        files_cosmo_train = [files_cosmo_train[i] for i in np.random.permutation(len(files_cosmo_train))]
        LOGGER.info(f"using cosmo tfrecords pipeline from {dirpath_cosmo} with {len(files_cosmo_train)} train files {len(files_cosmo_valid)} valid files")
        LOGGER.info(f"using astro tfrecords pipeline from {dirpath_astro} with {len(files_astro)} files")

        # prefetch size fix, the more the better, avoid autotune
        prefetch_size = 8

        # get generators

        # maps generators
        def get_gen_cosmo(files):

            from deep_lss.apps.run_recordmaker import get_parse_inverse_func
            par_inv = get_parse_inverse_func(n_fields=n_fields_per_survey, n_pix=img_size, n_maps=3, n_zbins=4, n_y=2) # TODO: update to avoid parameter hardcoding
            gen_cosmo = tf.data.TFRecordDataset(files, num_parallel_reads=20)
            gen_cosmo = gen_cosmo.map(par_inv, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
            gen_cosmo = gen_cosmo.batch(batch_size, drop_remainder=True)
            gen_cosmo = gen_cosmo.repeat()
            gen_cosmo = gen_cosmo.prefetch(prefetch_size)
            gen_cosmo = strategy.experimental_distribute_dataset(gen_cosmo)
            gen_cosmo = iter(gen_cosmo)
            return gen_cosmo

        gen_cosmo_train = get_gen_cosmo(files_cosmo_train)
        gen_cosmo_valid = get_gen_cosmo(files_cosmo_valid)

        # get astro generator
        gen_astro = tf.data.TFRecordDataset(files_astro, num_parallel_reads=1)
        gen_astro = gen_astro.map(parse_inverse_astro, num_parallel_calls=1, deterministic=True)
        gen_astro = gen_astro.cache()
        gen_astro = gen_astro.repeat()
        gen_astro = gen_astro.batch(batch_size, drop_remainder=True)
        gen_astro = strategy.experimental_distribute_dataset(gen_astro)
        gen_astro = iter(gen_astro)

        # get augmentations generator
        shuffles_len = int(1e6)
        np.random.seed(42)
        shuffles = np.array([np.random.permutation(ctx.n_fields_per_survey) for i in range(shuffles_len)])
        gen_shuff = tf.data.Dataset.from_tensor_slices(shuffles)
        gen_shuff = gen_shuff.batch(batch_size)
        gen_shuff = gen_shuff.repeat()
        gen_shuff = strategy.experimental_distribute_dataset(gen_shuff)
        gen_shuff = iter(gen_shuff)

        # processing datasets
        func_process_dataset = ctx.model_astro.get_fun_process_dataset(n_fields=ctx.n_fields_per_survey,
                                                                       n_pix=ctx.img_size,
                                                                       n_maps=ctx.n_maps,
                                                                       n_zbins=ctx.n_z_bins,
                                                                       batch_size=batch_size_split,
                                                                       transform_maps=ctx.training['transform_maps'])
        func_scale_to_prior_range = ctx.model_astro.get_func_scale_astro_to_prior_range()
        func_theta_transform_forward = ctx.model_astro.get_func_theta_transform_forward(method=ctx.training['transform_theta'])
        func_select_redshift_bins = ctx.select_redshift_bins

        # flip fields if using mosaic mode, otherwise it's not needed
        from deep_lss.networks import layers
        layer_flip = layers.FlipFields(n_fields=n_fields_per_survey)

        # process dataset pipeline
        @tf.function()
        def process(X, y_cosmo, y_astro, i_shuff, n_gal_per_pix):
            y_astro = func_scale_to_prior_range(y_astro)
            y = tf.concat((y_cosmo, y_astro), axis=1)
            y = tf.ensure_shape(y, shape=(batch_size_split, ctx.n_theta))
            X = func_process_dataset(X, y, n_gal_per_pix)

            # shuffle order inside mosaic if needed
            X = layer_flip(tf.gather(X, i_shuff, axis=1, batch_dims=1))
            y_trans = func_theta_transform_forward(y)
            return X, y_trans   

        # define training step
        @tf.function()
        def train_step(n_gal_per_pix, model, output_selects, opt, X, y_cosmo, i_cosmo, y_astro, i_shuff):

            # process dataset
            X, y = process(X, y_cosmo, y_astro, i_shuff, n_gal_per_pix)

            # custom training
            with tf.GradientTape(persistent=True) as g:

                # predictions for multimodel 
                loss_total, loss_parts, err = model(inputs=(X, y), training=True)
                loss_total = [loss/strategy.num_replicas_in_sync for loss in loss_total]

            # get and apply grads
            # gradients = g.gradient(loss_multi, model.trainable_variables)
            for n in range(n_nets):
                # divide by number of replicas, follow:
                # https://www.tensorflow.org/tutorials/distribute/custom_training#define_the_loss_function
                # they dont' recommend using the mean, but it should be ok given that the batch size is always fixed
                gradients = g.gradient(loss_total[n], model.trainable_variables)
                opt[n].apply_gradients([(grad, var) for (grad, var) in zip(gradients, model.trainable_variables) if grad is not None])
            
            return tf.stack(loss_parts)

        @tf.function
        def distributed_train_step(n_gal_per_pix, model, output_selects, opt):

            # get data from generators
            X, y_cosmo, i_cosmo = gen_cosmo_train.get_next()
            y_astro = gen_astro.get_next()
            i_shuff = gen_shuff.get_next()

            # run strategy
            loss = strategy.run(train_step, args=(n_gal_per_pix, model, output_selects, opt, X, y_cosmo, i_cosmo, y_astro, i_shuff))

            # compute combined loss from distributed parts
            tot_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
            if test:
                tf.print('tot_loss', tot_loss)
        
            return tot_loss

        # define test step
        @tf.function()
        def valid_step(n_gal_per_pix, model, output_selects, opt, X, y_cosmo, i_cosmo, y_astro, i_shuff):

            # process dataset
            X, y = process(X, y_cosmo, y_astro, i_shuff, n_gal_per_pix)

            # predictions for multimodel 
            loss_total, loss_parts, err = model(inputs=(X, y), training=False)

            return tf.stack(loss_parts)

        @tf.function
        def distributed_valid_step(n_gal_per_pix, model, output_selects, opt):

            # get data from generators
            X, y_cosmo, i_cosmo = gen_cosmo_valid.get_next()
            y_astro = gen_astro.get_next()
            i_shuff = gen_shuff.get_next()

            # run strategy
            loss = strategy.run(valid_step, args=(n_gal_per_pix, model, output_selects, opt, X, y_cosmo, i_cosmo, y_astro, i_shuff))

            # compute combined loss from distributed parts
            tot_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
            if test:
                tf.print('tot_loss_valid', tot_loss)
        
            return tot_loss

        # get number of galaxies per pixel depending on epoch
        def decay(i, n0, half_time):
            return  n0*np.exp(-(np.log(2.)/half_time)*i)

        def get_n_gal_per_pix(j, n_eff_target):
            n_gal_per_pix_target = n_eff_target * ctx.model_astro.pixel_area_arcmin2
            n_gal_per_pix = n_gal_per_pix_target + decay(j, ctx.training['neff_decay_n0'], ctx.training['neff_decay_half_time'])
            return tf.cast(n_gal_per_pix, dtype=tf.float32)

        # example image storing function
        def store_example_images(n_batches=1):

            for i in range(n_batches):

                X, y_cosmo, i_cosmo = gen_cosmo_valid.get_next()
                y_astro = gen_astro.get_next()
                i_shuff = gen_shuff.get_next()
                X, y = process(X, y_cosmo, y_astro, i_shuff, n_gal_per_pix=ctx.model_astro.ng_eff)
                for net in nets:
                    data_example = {'y':y}
                    if hasattr(net, 'network_imgs'):
                        data_example['imgs'] = net.network_imgs(X, training=False)
                    if hasattr(net, 'network_psds'):
                        data_example['psds'] = net.network_psds(X, training=False)
                    utils_io.store_example_images(dir_output, data_example, tag=f'{net.netname}_batch{i}')

        # store examples
        if test:
            store_example_images(n_batches=1)

        # benchmark readout when running in main mode/benchmark
        if not test:
    
            n_fields_total = 57*12*1000
            n_batches_benchmark = n_fields_total//n_fields_per_survey//batch_size # full dataset
            LOGGER.info(f'running readout benchmark with n_batches_benchmark={n_batches_benchmark}')

            @tf.function()
            def run_benchmark_readout():

                for i in range(n_batches_benchmark):
                    X, y_cosmo, i_cosmo = gen_cosmo_train.get_next()

            time_start = time.time()
            run_benchmark_readout()
            time_readout = time.time() - time_start
            LOGGER.info(f'readout benchmark time={time_readout/60:2.5f} time_per_field={time_readout/n_fields_total*1000:2.4f}ms')

        # epoch loop
        for j in range(n_epochs):

            # get most recent step and current noise level
            step_no = nets[0].meta_info['step'] 
            n_gal_per_pix_current = get_n_gal_per_pix(step_no, n_eff_target=ctx.model_astro.ng_eff)

            # verb and time mesurement
            LOGGER.critical(f'======== epoch {j} {ctx.tag} running {n_batches} batches step_no={step_no} n_gal_per_pix_current={n_gal_per_pix_current:10.2f} -> {ctx.model_astro.n_gal_per_pix:10.2f}')
            LOGGER.timer.start('epoch')
            time_start_epoch = time.time()

            # trianing loop
            epoch_loss = []
            epoch_loss_valid = {}
            epoch_ngal = []
            for i in range(n_batches):

                # noise level augmentation
                n_gal_per_pix_current = get_n_gal_per_pix(step_no+i, n_eff_target=ctx.model_astro.ng_eff)

                # this is the main training step
                loss = distributed_train_step(n_gal_per_pix=n_gal_per_pix_current, 
                                              output_selects=output_selects,
                                              model=multinet.loss,
                                              opt=[net.opt for net in nets])

                # store losses for later logging
                epoch_loss.append(loss)
                # epoch_errs.append(errs)
                epoch_ngal.append(n_gal_per_pix_current)

                # this is the main training step
                if i % valid_freq == 0:
                    loss_valid = distributed_valid_step(n_gal_per_pix=n_gal_per_pix_current, 
                                                        output_selects=output_selects,
                                                        model=multinet.loss,
                                                        opt=[net.opt for net in nets])
                    epoch_loss_valid[step_no+i] = loss_valid


            # report times
            time_per_field = (time.time()-time_start_epoch)/batch_size/n_batches/n_fields_per_survey*1000 # ms
            time_per_field_per_net = time_per_field/len(nets)
            LOGGER.info(f'n_batches={n_batches} time_per_field={time_per_field:2.4f}ms time_per_field_per_net={time_per_field_per_net:2.4f}ms')

            # write logs
            LOGGER.info('writing logs')

            # training loss
            for i in range(n_batches):  
                multinet.meta_info['step'] += 1     
                for n in range(n_nets): 
                    nets[n].meta_info['step'] += 1
                    with nets[n].writer.as_default():
                        tf.summary.scalar('loss_total', epoch_loss[i][n,0], step=tf.cast(nets[n].meta_info['step'], tf.int64))
                        tf.summary.scalar('loss_part1', epoch_loss[i][n,1], step=tf.cast(nets[n].meta_info['step'], tf.int64))
                        tf.summary.scalar('loss_part2', epoch_loss[i][n,2], step=tf.cast(nets[n].meta_info['step'], tf.int64))
                        tf.summary.scalar('ngal_per_pix', epoch_ngal[i], step=tf.cast(nets[n].meta_info['step'], tf.int64))
            
            # validation loss
            for i in epoch_loss_valid:
                for n in range(n_nets): 
                    with nets[n].writer.as_default():
                        tf.summary.scalar('loss_total_valid', epoch_loss_valid[i][n,0], step=tf.cast(i, tf.int64))

            # up the epoch
            for n in range(n_nets): 
                nets[n].meta_info['epoch'] += 1

            # store checkpoints
            store_checkpoints(nets, dirpath_checkpoints)

            # check if time is up
            LOGGER.critical(f"tag={ctx.tag} epoch={j} time={LOGGER.timer.elapsed('epoch')}")
            time_elapsed = (time.time()-time_start_job)/60.
            if (walltime - time_elapsed) < time_cleanup:
                LOGGER.info(f'done {j} epoch, time out {time_elapsed:2.1f} / {walltime:2.1f}')
                return
            else:
                LOGGER.info(f'done {j} epoch, time_elapsed={time_elapsed:2.1f} walltime={walltime:2.1f} time_cleanup={time_cleanup:2.1f}')


def store_checkpoints(nets, dirpath_checkpoints):

    for net in nets:
        filepath_check = get_checkpoint_name(dirpath_checkpoints, epoch='last', tag=net.netname)
        net.save(filepath_check)


if __name__=='__main__':

    for i in main(indices=[0], args=sys.argv[1:]):
        pass












        # # define training step
        # @tf.function()
        # def dummy_train_step(n_gal_per_pix, model, output_selects, opt, X, y_cosmo, i_cosmo, y_astro, i_shuff):

        #     # get output containers
        #     losses_total = tf.TensorArray(dtype=tf.float32, size=n_nets, dynamic_size=False)
        #     losses_parts = tf.TensorArray(dtype=tf.float32, size=n_nets, dynamic_size=False)
        #     errs = tf.TensorArray(dtype=tf.float32, size=n_nets, dynamic_size=False)

        #     # process dataset
        #     X, y = process(X, y_cosmo, y_astro, i_shuff, n_gal_per_pix)

        #     for n in range(n_nets):

        #         losses_total = losses_total.write(n, 0.)
        #         losses_parts = losses_parts.write(n, [0., 1., 2.])
        #         errs = errs.write(n, 0.)

        #     loss_multi = -1.

        #     return loss_multi, losses_parts.stack(), errs.stack()

        # @tf.function
        # def dummy_distributed_train_step(n_gal_per_pix, model, output_selects, opt):

        #     # get data from generators
        #     X, y_cosmo, i_cosmo = gen_cosmo.get_next()
        #     y_astro = gen_astro.get_next()
        #     i_shuff = gen_shuff.get_next()

        #     loss, losses_parts, errs = strategy.run(dummy_train_step, args=(n_gal_per_pix, model, output_selects, opt, X, y_cosmo, i_cosmo, y_astro, i_shuff))

        #     return loss, losses_parts, errs




        # # example images
        # if test:
        # # if False:

        #     # op sequence

        #     X, y_cosmo, i_cosmo = gen_cosmo.get_next()
        #     y_astro = gen_astro.get_next()
        #     i_shuff = gen_shuff.get_next()
        #     X, y = process(X, y_cosmo, y_astro, i_shuff, n_gal_per_pix=1000)
        #     X = pipe(X)
        #     utils_io.store_example_images(dir_output, data_example=(X, y), tag='train_batch1')

        #     X, y_cosmo, i_cosmo = gen_cosmo.get_next()
        #     y_astro = gen_astro.get_next()
        #     i_shuff = gen_shuff.get_next()
        #     X, y = process(X, y_cosmo, y_astro, i_shuff, n_gal_per_pix=1000)
        #     X = pipe(X)
        #     utils_io.store_example_images(dir_output, data_example=(X, y), tag='train_batch2')
