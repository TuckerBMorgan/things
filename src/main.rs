use ndarray::prelude::*;
use tsuga::prelude::*;
use euphoria::prelude::*;
use gymnasium::prelude::*;

use ndarray_stats::QuantileExt;
use rand::prelude::*;

fn one_hot_embedding(state: usize, number_of_states: usize) -> Array2<f32> {
    let mut embdeded_state = Array2::zeros((1, number_of_states));
    embdeded_state[[0, state]] = 1.0f32;
    return embdeded_state;
}

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
    let mut cartpole = make(&String::from("Cartpole")).unwrap();
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