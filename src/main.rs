use ndarray::prelude::*;
use tsuga::prelude::*;

use minifb::{Key, ScaleMode, Window, WindowOptions};
use mnist::*;
use ndarray_stats::QuantileExt;
use rand::prelude::*;
use std::collections::HashMap;
const LABELS: &[&'static str] = &["0 ", "1 ", "2 ", "3 ", "4 ", "5 ", "6 ", "7 ", "8 ", "9 "];


struct FrozenLake {
    pub transition_diagram: HashMap<usize, [usize;4]>,
    pub what_is_state: Vec<char>,
    pub state_type_to_reward: HashMap<char, usize>,
    pub current_state: usize
}

impl FrozenLake {
    pub fn new() -> FrozenLake {
        let transition_diagram: HashMap<usize, [usize;4]> = [
        //First Row
        (0,  [0, 1, 4, 0]),
        (1,  [1, 2, 5, 0]),
        (2,  [2, 3, 6, 1]),
        (3,  [3, 3, 7, 2]),
        //Second Row
        (4,  [0, 5, 8, 4]),
        (5,  [1, 6, 9, 4]),
        (6,  [2, 7, 10, 5]),
        (7,  [3, 7, 11, 6]),

        (8,  [4, 9,  12, 8]),
        (9,  [5, 10, 13, 8]),
        (10, [6, 11, 14, 9]),
        (11, [7, 11, 15, 10]),

        (12,  [8,  13,  12, 12]),
        (13,  [9,  14,  13, 8]),
        (14,  [10, 15,  14, 9]),
        (15,  [15, 15,  15, 15]),
        ].iter().cloned().collect();

        let what_is_state = vec!['S', 'F', 'F', 'F',
                             'F', 'H', 'F', 'H',
                             'F', 'F', 'F', 'H',
                             'H', 'F', 'F', 'G'];

        let state_type_to_reward : HashMap<char, usize> = [('S', 0), ('F', 0), ('H', 0), ('G', 1)].iter().cloned().collect();

        FrozenLake {transition_diagram, what_is_state, state_type_to_reward, current_state: 0}
    }

    pub fn reset(&mut self) -> usize {
        self.current_state = 0;
        self.current_state
    }

    //Returns, (next_state, reward, is)
    pub fn step(&mut self, action: usize) -> Result<(usize, f32, bool), &'static str> {
        if action > 3 {
            return Err("bad Action");
        }

        let mut rng = rand::thread_rng();
        let num: usize = rng.gen_range(0, 100);

        //Thirty percent chance to not go where we wanted
        let picked_action;
        if num < 25 {
            let mut rng = rand::thread_rng();
            let num: usize = rng.gen_range(0, 4);
            picked_action = num;
        }
        else {
            picked_action = action;
        }

        self.current_state = self.transition_diagram[&self.current_state][picked_action];
        let is_done = self.current_state == 15 
                    || self.current_state == 5
                    || self.current_state == 7
                    || self.current_state == 11
                    || self.current_state == 12;
        return Ok((self.current_state, self.state_type_to_reward[&self.what_is_state[self.current_state]] as f32, is_done));
    }
}

fn max_reward_in_state(state: usize, state_rewards : &HashMap<usize, Vec<ActionPair>>) -> f32 {

    let action_rewards = &state_rewards[&state];
    let mut highest_reward = 0.0f32;
    for i in action_rewards {
        if i.calcualte_action_value() > highest_reward {
            highest_reward = i.calcualte_action_value();
        }
    }

    return highest_reward;
}

fn run(state_rewards: &mut HashMap<usize, Vec<ActionPair>>, learn: bool, no_search: bool) -> f32 {
    let reward_decay = 0.95f32;
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
            let rewards_based_on_action : Vec<f32> = state_rewards[&state].iter().map(|x|x.calcualte_action_value()).collect();
            let mut highest_reward = 0.0f32;
            let mut best_index = 0;
            for (i, rewards) in rewards_based_on_action.iter().enumerate() {
                if *rewards > highest_reward {
                    highest_reward = *rewards;
                    best_index = i;
                }
            }
            next_action = best_index;
        }
        //but 5% of time follow a random policy
        else {
            next_action = rng.gen_range(0, 4);
        }
        let result = fl.step(next_action);
        match result {
            Ok((next_state, reward, is_done)) => {                
                if learn {
                    if reward == 1.0f32 {
                        //HORRIBLE HORRIBLE HORRIBLE HACK TO DEAL WITH TERMINAL REWARD STATE
                        for i in 0..4 {
                            state_rewards.get_mut(&next_state).unwrap()[i].add_reward(reward);
                        }
                    }

                    let state_value = reward  + (reward_decay * max_reward_in_state(next_state, &state_rewards));
                    let current_value = state_rewards.get_mut(&state).unwrap()[next_action].calcualte_action_value();
                    state_rewards.get_mut(&state).unwrap()[next_action].add_reward(state_value - current_value);
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

#[derive(Debug)]
pub struct ActionPair {
    pub running_value: f32,
    pub number_of_values: usize
}

impl ActionPair {
    pub fn new() -> ActionPair {
        ActionPair {
            running_value: 0.0f32,
            number_of_values: 0
        }
    }
    
    pub fn add_reward(&mut self, reward: f32) {
        if self.number_of_values == 0 {
            self.running_value = reward;
            self.number_of_values = 1;
        }
        else {
            self.number_of_values += 1;
            self.running_value += reward / (self.number_of_values as f32);
        }
    }

    pub fn calcualte_action_value(&self) -> f32 {
        return self.running_value;
    }
}

fn frozen_lake() {

    let mut state_rewards : HashMap<usize, Vec<ActionPair>> = HashMap::new();
    for i in 0..16 {
        let action_rewards = vec![ActionPair::new(), ActionPair::new(), ActionPair::new(), ActionPair::new()];
        state_rewards.insert(i, action_rewards);
    }

 
    for i in 0..10000 {
        let _ = run(&mut state_rewards, true, false);
        if i % 100 == 0 {
            let mut total_run_reward = 0.0f32;
            for _ in 0..100 {
                let run_reward = run(&mut state_rewards, false, true);
                total_run_reward += run_reward;            
            }
            println!("Average run reward was {}", total_run_reward / 100.0f32);
        }
    }
    println!("up: {}", state_rewards[&1][0].calcualte_action_value());
    println!("right: {}", state_rewards[&1][1].calcualte_action_value());
    println!("down: {}", state_rewards[&1][2].calcualte_action_value());
    println!("left: {}", state_rewards[&1][3].calcualte_action_value());

    println!("up: {}", state_rewards[&15][0].calcualte_action_value());
    println!("right: {}", state_rewards[&15][1].calcualte_action_value());
    println!("down: {}", state_rewards[&15][2].calcualte_action_value());
    println!("left: {}", state_rewards[&15][3].calcualte_action_value());

    //println!("{:?}", state_rewards)
}


fn main() {
    frozen_lake();
    return;
    let (input, output, test_input, test_output) = mnist_as_ndarray();
    println!("Successfully unpacked the MNIST dataset into Array2<f32> format!");

    // Let's see an example of the parsed MNIST dataset on both the training and testing data
    let mut rng = rand::thread_rng();
    let mut num: usize = rng.gen_range(0, input.nrows());

    println!(
        "Input record #{} has a label of {}",
        num,
        output.slice(s![num, ..])
    );
    display_img(input.slice(s![num, ..]).to_owned());

    num = rng.gen_range(0, test_input.nrows());
    println!(
        "Test record #{} has a label of {}",
        num,
        test_output.slice(s![num, ..])
    );
    display_img(test_input.slice(s![num, ..]).to_owned());

    // Now we can begin configuring any additional hidden layers, specifying their size and activation function
    let mut layers_cfg: Vec<FCLayer> = Vec::new();
    let sigmoid_layer_0 = FCLayer::new("sigmoid", 128);
    layers_cfg.push(sigmoid_layer_0);
    let sigmoid_layer_1 = FCLayer::new("sigmoid", 64);
    layers_cfg.push(sigmoid_layer_1);

    // The network can now be built using the specified layer configurations
    // Several other options for tuning the network's performance are available as well
    let mut fcn = FullyConnectedNetwork::default(input, output)
        .add_layers(layers_cfg)
        .iterations(10_000)
        .min_iterations(700)
        .error_threshold(0.05)
        .learnrate(0.01)
        .batch_size(200)
        .validation_pct(0.0001)
        .build();

    // Training occurs in place on the network
    fcn.train().expect("An error occurred while training");

    // We can now pass an appropriately-sized input through our trained network,
    // receiving an Array2<f32> on the output
    let test_result = fcn.evaluate(test_input.clone());

    // And will compare that output against the ideal one-hot encoded testing label array
    compare_results(test_result.clone(), test_output);

    // Now display a singular value with the classification spread to see an example of the actual values
    num = rng.gen_range(0, test_input.nrows());
    println!(
        "Test result #{} has a classification spread of:\n------------------------------",
        num
    );
    for i in 0..LABELS.len() {
        println!("{}: {:.2}%", LABELS[i], test_result[[num, i]] * 100.);
    }
    display_img(test_input.slice(s![num, ..]).to_owned());
}

fn mnist_as_ndarray() -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let (trn_size, _rows, _cols) = (60_000, 28, 28);
    let tst_size = 10_000;

    // Deconstruct the returned Mnist struct.
    // You can see the default Mnist struct at https://docs.rs/mnist/0.4.0/mnist/struct.MnistBuilder.html
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path("data/mnist")
        .label_format_one_hot()
        .download_and_extract()
        .finalize();

    // Convert the returned Mnist struct to Array2 format
    let trn_lbl: Array2<f32> = Array2::from_shape_vec((trn_size, 10), trn_lbl)
        .expect("Error converting labels to Array2 struct")
        .map(|x| *x as f32);
    // println!("The first digit is a {:?}",trn_lbl.slice(s![image_num, ..]) );

    // Can use an Array2 or Array3 here (Array3 for visualization)
    let trn_img = Array2::from_shape_vec((trn_size, 784), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);
    // println!("{:#.0}\n",trn_img.slice(s![image_num, .., ..]));

    // Convert the returned Mnist struct to Array2 format
    let tst_lbl: Array2<f32> = Array2::from_shape_vec((tst_size, 10), tst_lbl)
        .expect("Error converting labels to Array2 struct")
        .map(|x| *x as f32);
    println!("tst {:?}", tst_lbl);

    let tst_img = Array2::from_shape_vec((tst_size, 784), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 256.);

    (trn_img, trn_lbl, tst_img, tst_lbl)
}

fn compare_results(actual: Array2<f32>, ideal: Array2<f32>) {
    let mut correct_number = 0;
    for i in 0..actual.nrows() {
        let result_row = actual.slice(s![i, ..]);
        let output_row = ideal.slice(s![i, ..]);

        if result_row.argmax() == output_row.argmax() {
            correct_number += 1;
        }
    }
    println!(
        "Total correct values: {}/{}, or {}%",
        correct_number,
        actual.nrows(),
        (correct_number as f32) * 100. / (actual.nrows() as f32)
    );
}

// Displays in an MNIST image in a pop-up window
fn display_img(input: Array1<f32>) {
    let img_vec: Vec<u8> = input.to_vec().iter().map(|x| (*x * 256.) as u8).collect();
    // println!("img_vec: {:?}",img_vec);
    let mut buffer: Vec<u32> = Vec::with_capacity(28 * 28);
    for px in 0..784 {
        let temp: [u8; 4] = [img_vec[px], img_vec[px], img_vec[px], 255u8];
        // println!("temp: {:?}",temp);
        buffer.push(u32::from_le_bytes(temp));
    }

    let (window_width, window_height) = (600, 600);
    let mut window = Window::new(
        "Test - ESC to exit",
        window_width,
        window_height,
        WindowOptions {
            resize: true,
            scale_mode: ScaleMode::Center,
            ..WindowOptions::default()
        },
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // Limit to max ~60 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    while window.is_open() && !window.is_key_down(Key::Escape) && !window.is_key_down(Key::Q) {
        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window.update_with_buffer(&buffer, 28, 28).unwrap();
    }
}