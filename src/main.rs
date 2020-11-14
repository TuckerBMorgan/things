use ndarray::prelude::*;
use tsuga::prelude::*;
use euphoria::prelude::*;
use gymnasium::prelude::*;

use ndarray_stats::QuantileExt;
use rand::prelude::*;

/*
fn optimize_model(replay_buffer: &ReplayBuffer, model: &mut FullyConnectedNetwork, number_of_samples: usize) {
    let reward_discount_factor = 0.95f32;
    let memories = replay_buffer.sample_batch(number_of_samples);
    for memory in memories {
        let mut rewards_of_actions = model.predict(memory.0.clone());
        let mut rewards_of_possible_next_state = model.predict(memory.3.clone());
        if *memory.4 == true {
            rewards_of_actions[[0, *memory.1]] = *memory.2;
        } else {
            rewards_of_actions[[0, *memory.1]] = *memory.2 + (reward_discount_factor * rewards_of_possible_next_state.max().unwrap());
        }
        model.single_training_batch(memory.0.clone(), rewards_of_actions, 1);
    }
}
*/

fn one_hot_embedding(state: usize, number_of_states: usize) -> Array2<f32> {
    let mut embdeded_state = Array2::zeros((1, number_of_states));
    embdeded_state[[0, state]] = 1.0f32;
    return embdeded_state;
}
/*
fn run_but_with_networks(model: &mut FullyConnectedNetwork, remember: bool, no_search: bool, replay_buffer: &mut ReplayBuffer) -> f32 {
    let mut fl = FrozenLake::new();
    let mut state = fl.reset();
    let mut done = false;
    let mut total_run = 0.0;
    while !done {
        let mut rng = rand::thread_rng();
        let num: usize = rng.gen_range(0, 100);
        let next_action;
        //Most of the time, follow a greedy policy 
        if no_search == true || num < 90 {
            let embbded_state = one_hot_embedding(state, 16);
            let action_values = model.predict(embbded_state);
            let mut larget_index = 0;
            let mut larget_value = action_values[[0, 0]];
            for i in 0..4 {
                if action_values[[0, i]] > larget_value {
                    larget_index = i;
                    larget_value = action_values[[0, i]];
                }
            }
            next_action = larget_index;
        }
        //but 5% of time follow a random policy
        else {
            next_action = rng.gen_range(0, 4);
        }
        let result = fl.step(next_action);
        match result {
            Ok((next_state, reward, is_done)) => {                
                if remember {
                    replay_buffer.add_memory(one_hot_embedding(state, 16), next_action, reward, one_hot_embedding(next_state, 16), done);
                }
                state = next_state;
                done = is_done;
                total_run += reward;
            },
            Err(_) => {}
        }
    }
    return total_run;
}
*/

#[inline]
fn highest_index(array: &Array2<f32>) -> usize {
    let mut highest_value = f32::MIN;
    let mut index = 0;
    for i in 0..array.ncols() {        
        if array[[0, i]] > highest_value {
            highest_value = array[[0, i]];
            index = i;
        }
    }
    return index;
}

fn single_run_for_cartpole(networks: &mut TargetNetworkDQN, remember: bool, follow_random: bool, number_of_allowed_actions: usize, e_barrior: usize) -> f32 { 
    let mut cartpole = Cartpole::new();
    let mut done = false;
    let mut total_reward = 0.0f32;
    let mut state = cartpole.reset();
    let mut number_of_taken_actions = 0;
    while !done && number_of_taken_actions < number_of_allowed_actions {

        let mut rng = rand::thread_rng();
        let should_random_act = rng.gen_range(0, 100);
        let action_to_take;
        if should_random_act < e_barrior && follow_random == true {
            action_to_take = rng.gen_range(0, 2);
        }
        else {
            let action_values = networks.predict(state.clone());
            action_to_take = highest_index(&action_values);
            /*
            if follow_random == follow_random {
                println!("{}", action_values);
                println!("{}", action_to_take);
            }
            */
        }
        let (next_state, mut reward, is_done) = cartpole.step(action_to_take);
        total_reward += reward;
        if remember {
            if is_done == true {
                reward = -10.0f32;                
            }
            let memory = GenericMemory::new(state.clone(), action_to_take, reward, next_state.clone(), is_done);
            networks.add_memory(memory);
        }
        done = is_done;
        state = next_state;
        number_of_taken_actions += 1;
    }

    return total_reward;
}

fn cartpole_env(model: FullyConnectedNetwork, model_t:FullyConnectedNetwork) {
    let mut tndqn = TargetNetworkDQN::new(model, model_t);
    let mut number_of_valdiation_runs = 0;
    for i in 0..100000 {
        let e = ((10000 as f32 - i as f32) / 10000 as f32) * 100.0f32;
        single_run_for_cartpole(&mut tndqn, true, true, 200, e as usize + 1);
        if i % 10 == 0 && tndqn.number_of_collected_memories() > 32 {
            tndqn.optimize_models(32);
        }
        if i != 0 && i % 100 == 0 {
            number_of_valdiation_runs += 1;            
            let mut rewards_summed = 0.0f32;
            for _ in 0..100 {
                rewards_summed += single_run_for_cartpole(&mut tndqn, false, false, 200, 100);
            }

            println!("For run {}, avg reward was {}", number_of_valdiation_runs, rewards_summed / 100.0);
            tndqn.set_target_network_to_q_network();
        }
    }


}
/*
fn frozen_lake(model: &mut FullyConnectedNetwork, replay_buffer: &mut ReplayBuffer) {
    for i in 0..10000 {
        let _ = run_but_with_networks(model, true, false, replay_buffer);
        if i % 100 == 0 {
            let mut total_run_reward = 0.0f32;
            for _ in 0..100 {
                let run_reward = run_but_with_networks(model, false, true, replay_buffer);
                total_run_reward += run_reward;            
            }
            println!("Average run reward was {}", total_run_reward / 100.0f32);
        }
        if  replay_buffer.current_number_of_memories > 64 {
            optimize_model(&replay_buffer, model, 64);
        }
    }

    let mut array = Array2::zeros((16, 16));
    for i in 0..16 {
        array[[i, i]] = 1.0f32;
    }

    let value = model.predict(array);
    println!("{:?}", value);
}
*/

fn main() {
    //frozen_lake();

    // Let's see an example of the parsed MNIST dataset on both the training and testing data
    let mut rng = rand::thread_rng();
    let mut layers_cfg: Vec<FCLayer> = Vec::new();
    
    let sigmoid_layer_0 = FCLayer::new("relu", 4);
    layers_cfg.push(sigmoid_layer_0);
    layers_cfg.push(FCLayer::new("relu", 128));
    layers_cfg.push(FCLayer::new("relu", 128));
    layers_cfg.push(FCLayer::new("linear", 2));
    //let mut replay_buffer = ReplayBuffer::new(10000);

    let input = Array2::<f32>::zeros((128, 4));
    let output = Array2::<f32>::zeros((128, 2));
    // The network can now be built using the specified layer configurations
    // Several other options for tuning the network's performance are available as well
    let mut fcn = FullyConnectedNetwork::default(input, output)
        .add_layers(layers_cfg)
        .iterations(10_000)
        .min_iterations(700)
        .error_threshold(0.05)
        .learnrate(0.01)
        .batch_size(128)
        .validation_pct(0.0001)
        .build();

    let mut layers_cfg: Vec<FCLayer> = Vec::new();

    let sigmoid_layer_0 = FCLayer::new("relu", 4);
    layers_cfg.push(sigmoid_layer_0);
    layers_cfg.push(FCLayer::new("relu", 128));
    layers_cfg.push(FCLayer::new("relu", 128));
    layers_cfg.push(FCLayer::new("linear", 2));
    //let mut replay_buffer = ReplayBuffer::new(10000);

    let input = Array2::<f32>::zeros((128, 4));
    let output = Array2::<f32>::zeros((128, 2));
    let mut target_fcn = FullyConnectedNetwork::default(input, output)
        .add_layers(layers_cfg)
        .iterations(10_000)
        .min_iterations(700)
        .error_threshold(0.05)
        .learnrate(0.01)
        .batch_size(128)
        .validation_pct(0.0001)
        .build();
    //frozen_lake(&mut fcn, &mut replay_buffer);
    cartpole_env(fcn, target_fcn);
}