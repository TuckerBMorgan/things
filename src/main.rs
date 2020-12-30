use ndarray::prelude::*;
use tsuga::prelude::*;
use euphoria::prelude::*;
use gymnasium::prelude::*;

use rand::prelude::*;

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

type RewardShapingFunction = fn(f32, bool, &Array2<f32>) -> f32;

pub struct RunConvig<'a> {
    pub name: String,
    pub enviroment: &'a mut Box<dyn Enviroment>,
    pub remember: bool, 
    pub follow_random_policy: bool, 
    pub number_of_allowed_actions: usize, 
    pub e_barrior: usize, 
    pub reward_shaping_function: RewardShapingFunction,
    pub should_render: bool
}

impl<'a> RunConvig<'a> {
    pub fn new( name: String,
                enviroment: &'a mut Box<dyn Enviroment>,
                remember: bool,
                follow_random_policy: bool,
                number_of_allowed_actions: usize,
                e_barrior: usize,
                reward_shaping_function: RewardShapingFunction,
                should_render: bool
        ) -> RunConvig<'a> {
            RunConvig {
                name,
                enviroment,
                remember,
                follow_random_policy,
                number_of_allowed_actions,
                e_barrior,
                reward_shaping_function,
                should_render
            }
    }
}

fn single_run_for_enviroment(runconfig: &mut RunConvig, networks: &mut DiscreteSpaceNetwork) -> f32 {
    let enviroment = &mut runconfig.enviroment;
    let mut done = false;
    let mut total_reward = 0.0f32;
    let mut state = enviroment.reset();
    let mut number_of_taken_actions = 0;
    while !done && number_of_taken_actions < runconfig.number_of_allowed_actions {

        let mut rng = rand::thread_rng();
        let should_random_act = rng.gen_range(0, 100);
        let action_to_take;
        if should_random_act < runconfig.e_barrior && runconfig.follow_random_policy == true {
            action_to_take = rng.gen_range(0, 2);
        }
        else {
            let action_values = networks.predict(state.clone());
            action_to_take = highest_index(&action_values);
        }
        let (next_state, mut reward, is_done) = enviroment.step(action_to_take);
        if runconfig.should_render {
            enviroment.render();
        }
        reward = (runconfig.reward_shaping_function)(reward, is_done, &next_state);
        if runconfig.remember {
            let memory = GenericMemory::new(state.clone(), action_to_take, reward, next_state.clone(), is_done);
            networks.add_memory(memory);
        }
        total_reward += reward;
        done = is_done;
        state = next_state;
        number_of_taken_actions += 1;
    }

    return total_reward;
}

fn acrobot_reward_shaping(_reward: f32, terminal: bool, _state: &Array2<f32>) -> f32 {
    if terminal == true {
        return 1.0f32;
    }    
    return 0.0f32;
}

fn acrobot_env(model: FullyConnectedNetwork, model_t:FullyConnectedNetwork) {
    let mut tndqn = DiscreteSpaceNetwork::new(model, model_t);
    let mut number_of_valdiation_runs = 0;
    let mut enviroment = make(&String::from("Acrobot")).unwrap();
    let mut runconvig = RunConvig::new(String::from("Acrobot"), &mut enviroment, true, true, 200, 100, acrobot_reward_shaping, false);
    for i in 0..100000 {
        let e = ((10000 as f32 - i as f32) / 10000 as f32) * 100.0f32;

        //Set the convig ready for a training run(remember)
        runconvig.remember = true;
        runconvig.follow_random_policy = true;
        runconvig.e_barrior = e as usize + 1;
        runconvig.number_of_allowed_actions = 200;

        single_run_for_enviroment(&mut runconvig, &mut tndqn);

        if i % 10 == 0 && tndqn.number_of_collected_memories() > 32 {
            tndqn.optimize_models(32);
        }
        if i != 0 && i % 100 == 0 {
            number_of_valdiation_runs += 1;            
            let mut rewards_summed = 0.0f32;
            for _ in 0..100 {
                //Change convif over to a evaluation run(no learning)
                runconvig.remember = false;
                runconvig.follow_random_policy = false;
                runconvig.e_barrior = 100;
                rewards_summed += single_run_for_enviroment(&mut runconvig, &mut tndqn);
            }

            println!("For run {}, avg reward was {}", number_of_valdiation_runs, rewards_summed / 100.0);
            //tndqn.set_target_network_to_q_network();
        }
    }
}

fn cartpole_reward_shaping(reward: f32, terminal: bool, _state: &Array2<f32>) -> f32 {
    if terminal {
        return -10.0f32;
    }
   return reward;
}

fn cartpole_env(model: FullyConnectedNetwork, model_t:FullyConnectedNetwork) {
    let mut tndqn = DiscreteSpaceNetwork::new(model, model_t);
    let mut number_of_valdiation_runs = 0;

    let mut enviroment = make(&String::from("Cartpole")).unwrap();
    let mut runconvig = RunConvig::new(String::from("Cartpole"), &mut enviroment, true, true, 200, 100, cartpole_reward_shaping, true);

    for i in 0..100000 {
        let e = ((10000 as f32 - i as f32) / 10000 as f32) * 100.0f32;

        runconvig.remember = true;
        runconvig.follow_random_policy = true;
        runconvig.e_barrior = e as usize + 1;
        runconvig.should_render = true;

        single_run_for_enviroment(&mut runconvig, &mut tndqn);

        if i % 10 == 0 && tndqn.number_of_collected_memories() > 32 {
            tndqn.optimize_models(32);
        }
        if i != 0 && i % 1000 == 0 {
            number_of_valdiation_runs += 1;            
            let mut rewards_summed = 0.0f32;
            runconvig.remember = false;
            runconvig.follow_random_policy = false;
            runconvig.e_barrior = 100;
            runconvig.should_render = true;
            for _ in 0..100 {                
                rewards_summed += single_run_for_enviroment(&mut runconvig, &mut tndqn);
            }

            println!("For run {}, avg reward was {}", number_of_valdiation_runs, rewards_summed / 100.0);
            tndqn.set_target_network_to_q_network();
        }
    }
}

fn acrobot() {
    let mut layers_cfg: Vec<FCLayer> = Vec::new();
    
    let sigmoid_layer_0 = FCLayer::new("relu", 6);
    layers_cfg.push(sigmoid_layer_0);
    layers_cfg.push(FCLayer::new("relu", 128));
    layers_cfg.push(FCLayer::new("relu", 128));
    layers_cfg.push(FCLayer::new("linear", 2));

    let input = Array2::<f32>::zeros((128, 6));
    let output = Array2::<f32>::zeros((128, 2));
    // The network can now be built using the specified layer configurations
    // Several other options for tuning the network's performance are available as well
    let fcn = FullyConnectedNetwork::default(input, output)
        .add_layers(layers_cfg)
        .iterations(10_000)
        .min_iterations(700)
        .error_threshold(0.05)
        .learnrate(0.01)
        .batch_size(128)
        .validation_pct(0.0001)
        .build();

    let mut layers_cfg: Vec<FCLayer> = Vec::new();

    let sigmoid_layer_0 = FCLayer::new("relu", 6);
    layers_cfg.push(sigmoid_layer_0);
    layers_cfg.push(FCLayer::new("relu", 128));
    layers_cfg.push(FCLayer::new("relu", 128));
    layers_cfg.push(FCLayer::new("linear", 2));

    let input = Array2::<f32>::zeros((128, 6));
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
    fcn.blind_copy(&mut target_fcn);
    acrobot_env(fcn, target_fcn);
}

fn cartpole() {
    //frozen_lake();

    // Let's see an example of the parsed MNIST dataset on both the training and testing data
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
    let fcn = FullyConnectedNetwork::default(input, output)
        .add_layers(layers_cfg)
        .iterations(10_000)
        .min_iterations(700)
        .error_threshold(0.05)
        .learnrate(0.01)
        .batch_size(128)
        .validation_pct(0.001)
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
        .validation_pct(0.001)
        .build();
    fcn.blind_copy(&mut target_fcn);
    cartpole_env(fcn, target_fcn);
}

fn main() {
    let mut acrobot_thjing = make(&String::from("Acrobot")).unwrap();
    acrobot_thjing.reset();
    loop {
        acrobot_thjing.render();
    }
    cartpole();
    acrobot();

}