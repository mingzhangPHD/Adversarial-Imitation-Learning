import os
import shutil
from tensorboardX import SummaryWriter
import torch
import numpy as np
from collections import deque
import time
from ..tools import utils, utli


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def gailLearning_mujoco(cl_args, envs, envs_eval, actor_critic, agent, discriminator, rollouts, gail_train_loader, device, utli):

    log_save_name = utli.Log_save_name4gail(cl_args)
    log_save_path = os.path.join("./runs", log_save_name)
    if os.path.exists(log_save_path):
        shutil.rmtree(log_save_path)
    utli.writer = SummaryWriter(log_save_path)

    # model_dir = utli.Save_model_dir(cl_args.algo_id, cl_args.env_name)



    # Evaluate the initial network
    evaluations = []
    # begin optimize

    reward_window4Evaluate = deque(maxlen=10)
    time_step = 0
    episode_t = 0
    episode_timesteps = 0
    count = 0

    # begin optimize

    nsteps = cl_args.nsteps
    S_time = time.time()

    nenv = 1

    nbatch = np.floor(nsteps/nenv)
    nbatch = nbatch.astype(np.int16)
    nupdates = np.floor(cl_args.total_steps / nsteps)
    nupdates = nupdates.astype(np.int16)

    epinfobuf = deque(maxlen=10)

    # epgailbuf = deque(maxlen=10)

    episode_rewards = deque(maxlen=10)

    cum_gailrewards = .0

    i_update = 0

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    # write_result = utli.Write_Result(cl_args=cl_args)

    start = time.time()
    # num_updates = int(
    #     args.num_env_steps) // args.num_steps // args.num_processes

    while i_update < nupdates:

        episode_t += 1
        i_update += 1
        epinfos = []

        if cl_args.use_linear_lr_decay:
            # decrease learning rate linearly
            utli.update_linear_schedule(
                agent.optimizer, i_update, nupdates,
                cl_args.lr)


        for step in range(nbatch):
            time_step += 1

            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # action, log_prob, value = model_step(cl_args=cl_args, model=model, state=cur_state, device=device)
            # time.sleep(.002)
            # next_state, reward, done, infos = envs.step(action)   # error 01
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)


        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        # gail
        if cl_args.gail:
            if i_update >= 10:
                envs.venv.eval()
            # gail_epoch = args.gail_epoch
            gail_epoch = cl_args.gail_epoch
            if i_update < 10:
                gail_epoch = 100  # Warm up

            dis_losses, dis_gps, dis_entropys, dis_total_losses = [], [], [], []
            for _ in range(gail_epoch):
                # dis_loss, dis_gp, dis_entropy, dis_total_loss = \
                #     discriminator.update(replay_buf=rollouts, expert_buf=expert_buffer,
                #                          obsfilt=utils.get_vec_normalize(envs)._obfilt, batch_size=batch_size)
                dis_loss, dis_gp, dis_entropy, dis_total_loss = discriminator.update(gail_train_loader, rollouts, utils.get_vec_normalize(envs)._obfilt)
                dis_losses.append(dis_loss)
                dis_gps.append(dis_gp)
                dis_entropys.append(dis_entropy)
                dis_total_losses.append(dis_total_loss)

            utli.recordDisLossResults(results=(np.mean(np.array(dis_losses)),
                                               np.mean(np.array(dis_gps)),
                                               np.mean(np.array(dis_entropys)),
                                               np.mean(np.array(dis_total_losses))),
                                      time_step=time_step)
            # write_result.step_train(time_step)

            for step in range(nbatch):
                rollouts.rewards[step] = discriminator.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], cl_args.gamma,
                    rollouts.masks[step])

        # compute returns
        rollouts.compute_returns(next_value, cl_args.use_gae, cl_args.gamma,
                                 cl_args.gae_lambda, cl_args.use_proper_time_limits)

        # training PPO policy

        value_loss, action_loss, dist_entropy, total_loss = agent.update(rollouts)

        utli.recordLossResults(results=(value_loss,
                                        action_loss,
                                        dist_entropy,
                                        total_loss),
                               time_step=time_step)
        rollouts.after_update()


        epinfobuf.extend(epinfos)
        if not len(epinfobuf):
            continue
        eprewmean = safemean([epinfo['r'] for epinfo in epinfobuf])
        eplenmean = safemean([epinfo['l'] for epinfo in epinfobuf])

        utli.recordTrainResults(results=(eprewmean,
                                         eplenmean),
                                time_step=time_step)

        # write_result.step_train(time_step)

        print("Episode: %d,   Time steps: %d,   Mean length: %d    Mean Reward: %f  "
            % (episode_t, time_step, eplenmean, eprewmean))

        if i_update % cl_args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (i_update + 1) * cl_args.num_processes * cl_args.nsteps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(i_update, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

    E_time = time.time()
    # store results
    utli.store_results(evaluations, time_step, cl_args, S_time=S_time, E_time=E_time)

    return 0

def gailLearning_mujoco_test(cl_args, envs, envs_eval, actor_critic, agent, discriminator, rollouts, expert_buffer, device, utli):

    log_save_name = utli.Log_save_name4gail(cl_args)
    log_save_path = os.path.join("./runs", log_save_name)
    if os.path.exists(log_save_path):
        shutil.rmtree(log_save_path)
    utli.writer = SummaryWriter(log_save_path)

    # model_dir = utli.Save_model_dir(cl_args.algo_id, cl_args.env_name)



    # Evaluate the initial network
    evaluations = []
    # begin optimize

    reward_window4Evaluate = deque(maxlen=10)
    time_step = 0
    episode_t = 0
    episode_timesteps = 0
    count = 0

    # begin optimize

    nsteps = cl_args.nsteps
    S_time = time.time()

    nenv = 1

    nbatch = np.floor(nsteps/nenv)
    nbatch = nbatch.astype(np.int16)
    nupdates = np.floor(cl_args.total_steps / nsteps)
    nupdates = nupdates.astype(np.int16)

    epinfobuf = deque(maxlen=10)

    epgailbuf = deque(maxlen=10)

    episode_rewards = deque(maxlen=10)

    cum_gailrewards = .0

    i_update = 0
    dis_init = True

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    # write_result = utli.Write_Result(cl_args=cl_args)

    start = time.time()

    while i_update < nupdates:

        episode_t += 1
        i_update += 1
        epinfos = []

        if cl_args.use_linear_lr_decay:
            # decrease learning rate linearly
            utli.update_linear_schedule(
                agent.optimizer, i_update, nupdates,
                cl_args.lr)


        for step in range(nbatch):
            time_step += 1

            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # action, log_prob, value = model_step(cl_args=cl_args, model=model, state=cur_state, device=device)
            # time.sleep(.002)
            # next_state, reward, done, infos = envs.step(action)   # error 01
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)


        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        # gail
        if cl_args.gail:
            if i_update >= cl_args.gail_thre:
                envs.venv.eval()
            # gail_epoch = args.gail_epoch
            gail_epoch = cl_args.gail_epoch
            if i_update < cl_args.gail_thre:
                gail_epoch = cl_args.gail_pre_epoch  # Warm up

            dis_losses, dis_gps, dis_entropys, dis_total_losses = [], [], [], []
            for _ in range(gail_epoch):
                dis_loss, dis_gp, dis_entropy, dis_total_loss = \
                    discriminator.update_zm(replay_buf=rollouts, expert_buf=expert_buffer,
                                         obsfilt=utils.get_vec_normalize(envs)._obfilt, batch_size=cl_args.gail_batch_size)
                # dis_loss, dis_gp, dis_entropy, dis_total_loss = discriminator.update(gail_train_loader, rollouts, utils.get_vec_normalize(envs)._obfilt)
                dis_losses.append(dis_loss)
                dis_gps.append(dis_gp)
                dis_entropys.append(dis_entropy)
                dis_total_losses.append(dis_total_loss)

            if dis_init:
                utli.recordDisLossResults(results=(np.array(dis_losses)[0],
                                                   np.array(dis_gps)[0],
                                                   np.array(dis_entropys)[0],
                                                   np.array(dis_total_losses)[0]),
                                          time_step=0)
                dis_init = False


            utli.recordDisLossResults(results=(np.mean(np.array(dis_losses)),
                                               np.mean(np.array(dis_gps)),
                                               np.mean(np.array(dis_entropys)),
                                               np.mean(np.array(dis_total_losses))),
                                      time_step=time_step)
            # write_result.step_train(time_step)

            for step in range(nbatch):
                rollouts.rewards[step] = discriminator.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], cl_args.gamma,
                    rollouts.masks[step])
                if rollouts.masks[step] == 1:
                    cum_gailrewards += rollouts.rewards[step].item()
                else:
                    epgailbuf.append(cum_gailrewards)
                    cum_gailrewards=.0

        # compute returns
        rollouts.compute_returns(next_value, cl_args.use_gae, cl_args.gamma,
                                 cl_args.gae_lambda, cl_args.use_proper_time_limits)

        # training PPO policy

        value_loss, action_loss, dist_entropy, total_loss = agent.update(rollouts)

        utli.recordLossResults(results=(value_loss,
                                        action_loss,
                                        dist_entropy,
                                        total_loss),
                               time_step=time_step)
        rollouts.after_update()


        epinfobuf.extend(epinfos)
        if not len(epinfobuf):
            continue
        eprewmean = safemean([epinfo['r'] for epinfo in epinfobuf])
        eplenmean = safemean([epinfo['l'] for epinfo in epinfobuf])

        # utli.recordTrainResults(results=(eprewmean,
        #                                  eplenmean),
        #                         time_step=time_step)

        utli.recordTrainResults_gail(results=(eprewmean,
                                              eplenmean,
                                              np.mean(np.array(epgailbuf))),
                                time_step=time_step)

        # write_result.step_train(time_step)

        # print("Episode: %d,   Time steps: %d,   Mean length: %d    Mean Reward: %f  "
        #     % (episode_t, time_step, eplenmean, eprewmean))

        print("Episode: %d,   Time steps: %d,   Mean length: %d    Mean Reward: %f    Mean Gail Reward:%f"
            % (episode_t, time_step, eplenmean, eprewmean, np.mean(np.array(epgailbuf))))

        if i_update % cl_args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (i_update + 1) * cl_args.num_processes * cl_args.nsteps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(i_update, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

    E_time = time.time()
    # store results
    utli.store_results(evaluations, time_step, cl_args, S_time=S_time, E_time=E_time)

    return 0

def gailLearning_mujoco_origin(cl_args, envs, envs_eval, actor_critic, agent, discriminator, rollouts, gail_train_loader, device, utli):

    log_save_name = utli.Log_save_name4gail(cl_args)
    log_save_path = os.path.join("./runs", log_save_name)
    if os.path.exists(log_save_path):
        shutil.rmtree(log_save_path)
    utli.writer = SummaryWriter(log_save_path)

    # model_dir = utli.Save_model_dir(cl_args.algo_id, cl_args.env_name)



    # Evaluate the initial network
    evaluations = []
    # begin optimize

    reward_window4Evaluate = deque(maxlen=10)
    time_step = 0
    episode_t = 0
    episode_timesteps = 0
    count = 0

    # begin optimize

    nsteps = cl_args.num_steps
    S_time = time.time()

    nenv = 1

    nbatch = np.floor(nsteps/nenv)
    nbatch = nbatch.astype(np.int16)
    nupdates = np.floor(cl_args.num_env_steps / nsteps)
    nupdates = nupdates.astype(np.int16)

    epinfobuf = deque(maxlen=10)

    epgailbuf = deque(maxlen=10)

    episode_rewards = deque(maxlen=10)

    cum_gailrewards = .0

    i_update = 0
    dis_init = True

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    write_result = utli.Write_Result(cl_args=cl_args)

    start = time.time()

    while i_update < nupdates:

        episode_t += 1
        i_update += 1
        epinfos = []

        if cl_args.use_linear_lr_decay:
            # decrease learning rate linearly
            utli.update_linear_schedule(
                agent.optimizer, i_update, nupdates,
                cl_args.lr)

        for step in range(nbatch):
            time_step += 1

            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # action, log_prob, value = model_step(cl_args=cl_args, model=model, state=cur_state, device=device)
            # time.sleep(.002)
            # next_state, reward, done, infos = envs.step(action)   # error 01
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        # gail
        if cl_args.gail:
            if i_update >= cl_args.gail_thre:
                envs.venv.eval()
            # gail_epoch = args.gail_epoch
            gail_epoch = cl_args.gail_epoch
            if i_update < cl_args.gail_thre:
                gail_epoch = cl_args.gail_pre_epoch  # Warm up

            dis_losses, dis_gps, dis_entropys, dis_total_losses = [], [], [], []
            for _ in range(gail_epoch):
                # dis_loss, dis_gp, dis_entropy, dis_total_loss = \
                #     discriminator.update_zm(replay_buf=rollouts, expert_buf=expert_buffer,
                #                          obsfilt=utils.get_vec_normalize(envs)._obfilt, batch_size=cl_args.gail_batch_size)
                dis_loss, dis_gp, dis_entropy, dis_total_loss = discriminator.update(gail_train_loader, rollouts, utils.get_vec_normalize(envs)._obfilt)
                dis_losses.append(dis_loss)
                dis_gps.append(dis_gp)
                dis_entropys.append(dis_entropy)
                dis_total_losses.append(dis_total_loss)

            if dis_init:
                utli.recordDisLossResults(results=(np.array(dis_losses)[0],
                                                   np.array(dis_gps)[0],
                                                   np.array(dis_entropys)[0],
                                                   np.array(dis_total_losses)[0]),
                                          time_step=0)
                dis_init = False


            utli.recordDisLossResults(results=(np.mean(np.array(dis_losses)),
                                               np.mean(np.array(dis_gps)),
                                               np.mean(np.array(dis_entropys)),
                                               np.mean(np.array(dis_total_losses))),
                                      time_step=time_step)


            for step in range(nbatch):
                rollouts.rewards[step] = discriminator.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], cl_args.gamma,
                    rollouts.masks[step])
                if rollouts.masks[step] == 1:
                    cum_gailrewards += rollouts.rewards[step].item()
                else:
                    epgailbuf.append(cum_gailrewards)
                    cum_gailrewards=.0

        # compute returns
        rollouts.compute_returns(next_value, cl_args.use_gae, cl_args.gamma,
                                 cl_args.gae_lambda, cl_args.use_proper_time_limits)

        # training PPO policy

        value_loss, action_loss, dist_entropy, total_loss = agent.update(rollouts)

        utli.recordLossResults(results=(value_loss,
                                        action_loss,
                                        dist_entropy,
                                        total_loss),
                               time_step=time_step)
        rollouts.after_update()


        epinfobuf.extend(epinfos)
        if not len(epinfobuf):
            continue
        eprewmean = safemean([epinfo['r'] for epinfo in epinfobuf])
        eplenmean = safemean([epinfo['l'] for epinfo in epinfobuf])

        # utli.recordTrainResults(results=(eprewmean,
        #                                  eplenmean),
        #                         time_step=time_step)

        utli.recordTrainResults_gail(results=(eprewmean,
                                              eplenmean,
                                              np.mean(np.array(epgailbuf))),
                                time_step=time_step)

        write_result.step_train(time_step)


        print("Episode: %d,   Time steps: %d,   Mean length: %d    Mean Reward: %f    Mean Gail Reward:%f"
            % (episode_t, time_step, eplenmean, eprewmean, np.mean(np.array(epgailbuf))))

        if i_update % cl_args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (i_update + 1) * cl_args.num_processes * cl_args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(i_update, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

    E_time = time.time()
    # store results
    utli.store_results(evaluations, time_step, cl_args, S_time=S_time, E_time=E_time)

    return 0


def gailLearning_mujoco_BL(cl_args, envs, envs_eval, actor_critic, agent, discriminator, rollouts, expert_buffer, device, utli):

    log_save_name = utli.Log_save_name4gail(cl_args)
    log_save_path = os.path.join("./runs", log_save_name)
    if os.path.exists(log_save_path):
        shutil.rmtree(log_save_path)
    utli.writer = SummaryWriter(log_save_path)

    # model_dir = utli.Save_model_dir(cl_args.algo_id, cl_args.env_name)



    # Evaluate the initial network
    evaluations = []
    # begin optimize

    reward_window4Evaluate = deque(maxlen=10)
    time_step = 0
    episode_t = 0
    episode_timesteps = 0
    count = 0

    # begin optimize

    nsteps = cl_args.num_steps
    S_time = time.time()

    nenv = 1

    nbatch = np.floor(nsteps/nenv)
    nbatch = nbatch.astype(np.int16)
    nupdates = np.floor(cl_args.num_env_steps / nsteps)
    nupdates = nupdates.astype(np.int16)

    epinfobuf = deque(maxlen=10)

    epgailbuf = deque(maxlen=10)

    episode_rewards = deque(maxlen=10)

    cum_gailrewards = .0

    i_update = 0
    dis_init = True

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    write_result = utli.Write_Result(cl_args=cl_args)

    start = time.time()

    while i_update < nupdates:

        episode_t += 1
        i_update += 1
        epinfos = []

        if cl_args.use_linear_lr_decay:
            # decrease learning rate linearly
            utli.update_linear_schedule(
                agent.optimizer, i_update, nupdates,
                cl_args.lr)

        for step in range(nbatch):
            time_step += 1

            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # action, log_prob, value = model_step(cl_args=cl_args, model=model, state=cur_state, device=device)
            # time.sleep(.002)
            # next_state, reward, done, infos = envs.step(action)   # error 01
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        # gail
        if cl_args.gail:
            if i_update >= cl_args.gail_thre:
                envs.venv.eval()
            # gail_epoch = args.gail_epoch
            gail_epoch = cl_args.gail_epoch
            if i_update < cl_args.gail_thre:
                gail_epoch = cl_args.gail_pre_epoch  # Warm up

            dis_losses, dis_gps, dis_entropys, dis_total_losses = [], [], [], []
            for _ in range(gail_epoch):
                dis_loss, dis_gp, dis_entropy, dis_total_loss = \
                    discriminator.update_zm(replay_buf=rollouts, expert_buf=expert_buffer,
                                         obsfilt=utils.get_vec_normalize(envs)._obfilt, batch_size=cl_args.gail_batch_size)
                # dis_loss, dis_gp, dis_entropy, dis_total_loss = discriminator.update(gail_train_loader, rollouts, utils.get_vec_normalize(envs)._obfilt)
                dis_losses.append(dis_loss)
                dis_gps.append(dis_gp)
                dis_entropys.append(dis_entropy)
                dis_total_losses.append(dis_total_loss)

            if dis_init:
                utli.recordDisLossResults(results=(np.array(dis_losses)[0],
                                                   np.array(dis_gps)[0],
                                                   np.array(dis_entropys)[0],
                                                   np.array(dis_total_losses)[0]),
                                          time_step=0)
                dis_init = False


            utli.recordDisLossResults(results=(np.mean(np.array(dis_losses)),
                                               np.mean(np.array(dis_gps)),
                                               np.mean(np.array(dis_entropys)),
                                               np.mean(np.array(dis_total_losses))),
                                      time_step=time_step)


            for step in range(nbatch):
                rollouts.rewards[step] = discriminator.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], cl_args.gamma,
                    rollouts.masks[step])
                if rollouts.masks[step] == 1:
                    cum_gailrewards += rollouts.rewards[step].item()
                else:
                    epgailbuf.append(cum_gailrewards)
                    cum_gailrewards=.0

        # compute returns
        rollouts.compute_returns(next_value, cl_args.use_gae, cl_args.gamma,
                                 cl_args.gae_lambda, cl_args.use_proper_time_limits)

        # training PPO policy

        value_loss, action_loss, dist_entropy, total_loss = agent.update(rollouts)

        utli.recordLossResults(results=(value_loss,
                                        action_loss,
                                        dist_entropy,
                                        total_loss),
                               time_step=time_step)
        rollouts.after_update()


        epinfobuf.extend(epinfos)
        if not len(epinfobuf):
            continue
        eprewmean = safemean([epinfo['r'] for epinfo in epinfobuf])
        eplenmean = safemean([epinfo['l'] for epinfo in epinfobuf])

        # utli.recordTrainResults(results=(eprewmean,
        #                                  eplenmean),
        #                         time_step=time_step)

        utli.recordTrainResults_gail(results=(eprewmean,
                                              eplenmean,
                                              np.mean(np.array(epgailbuf))),
                                time_step=time_step)

        write_result.step_train(time_step)


        print("Episode: %d,   Time steps: %d,   Mean length: %d    Mean Reward: %f    Mean Gail Reward:%f"
            % (episode_t, time_step, eplenmean, eprewmean, np.mean(np.array(epgailbuf))))

        if i_update % cl_args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (i_update + 1) * cl_args.num_processes * cl_args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(i_update, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

    E_time = time.time()
    # store results
    utli.store_results(evaluations, time_step, cl_args, S_time=S_time, E_time=E_time)

    return 0

def Learning_process(args, envs, rollouts, actor_critic, agent, discr,gail_train_loader, device):


    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy, total_loss = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        # if (j % args.save_interval == 0
        #     or j == num_updates - 1) and args.save_dir != "":
        #     save_path = os.path.join(args.save_dir, args.algo)
        #     try:
        #         os.makedirs(save_path)
        #     except OSError:
        #         pass
        #
        #     torch.save([
        #         actor_critic,
        #         getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
        #     ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))

    return True

def Learning_process_record(args, envs, rollouts, actor_critic, agent, discr,gail_train_loader, device):


    log_save_name = utli.Log_save_name4gail(args)
    log_save_path = os.path.join("./runs", log_save_name)
    if os.path.exists(log_save_path):
        shutil.rmtree(log_save_path)
    utli.writer = SummaryWriter(log_save_path)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    epinfobuf = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    time_step = 0
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            time_step += 1
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfobuf.append(maybeepinfo)

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            dis_losses, dis_gps, dis_entropys, dis_total_losses = [], [], [], []
            for _ in range(gail_epoch):
                dis_loss, dis_gp, dis_entropy, dis_total_loss = discr.update(gail_train_loader, rollouts,
                                                                             utils.get_vec_normalize(envs)._obfilt)
                dis_losses.append(dis_loss)
                dis_gps.append(dis_gp)
                dis_entropys.append(dis_entropy)
                dis_total_losses.append(dis_total_loss)

            utli.recordDisLossResults(results=(np.mean(np.array(dis_losses)),
                                               np.mean(np.array(dis_gps)),
                                               np.mean(np.array(dis_entropys)),
                                               np.mean(np.array(dis_total_losses))),
                                      time_step=time_step)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy, total_loss = agent.update(rollouts)

        utli.recordLossResults(results=(value_loss,
                                        action_loss,
                                        dist_entropy,
                                        total_loss),
                               time_step=time_step)

        rollouts.after_update()

        # epinfobuf.extend(epinfos)
        if not len(epinfobuf):
            continue
        eprewmean = safemean([epinfo['r'] for epinfo in epinfobuf])
        eplenmean = safemean([epinfo['l'] for epinfo in epinfobuf])

        utli.recordTrainResults(results=(eprewmean,
                                         eplenmean),
                                time_step=time_step)

        # write_result.step_train(time_step)

        print("Episode: %d,   Time steps: %d,   Mean length: %d    Mean Reward: %f  "
            % (j, time_step, eplenmean, eprewmean))

        # save for every interval-th episode or for the last epoch
        # if (j % args.save_interval == 0
        #     or j == num_updates - 1) and args.save_dir != "":
        #     save_path = os.path.join(args.save_dir, args.algo)
        #     try:
        #         os.makedirs(save_path)
        #     except OSError:
        #         pass
        #
        #     torch.save([
        #         actor_critic,
        #         getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
        #     ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))

    return True