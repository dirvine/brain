use std::time::Duration;

#[derive(PartialOrd, PartialEq, Clone, Copy)]
pub struct Params {
    population_size: u32,
    time_alive_min: Duration,
    compat_threshold: f64,
    trait_mutate_prob: f64,
    trait_mutation_power: f64,
    survival_threshold: f64,
    mutate_only_prob: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            population_size: 150,
            time_alive_min: Duration::from_millis(10),
            compat_threshold: 3.3,
            trait_mutate_prob: 0.2,
            trait_mutation_power: 1.5,
            survival_threshold: 1.5,
            mutate_only_prob: 0.02,
        }
    }
}

impl Params {
    /// Getter
    pub fn population_size(&self) -> &u32 {
        &self.population_size
    }
    /// Getter
    pub fn time_alive_min(&self) -> &Duration {
        &self.time_alive_min
    }
    /// Getter
    pub fn compat_threshold(&self) -> &f64 {
        &self.compat_threshold
    }
    /// Getter
    pub fn trait_mutate_prob(&self) -> f64 {
        self.trait_mutate_prob
    }
    /// Getter
    pub fn trait_mutation_power(&self) -> f64 {
        self.trait_mutation_power
    }
    /// Getter
    pub fn survival_threshold(&self) -> &f64 {
        &self.survival_threshold
    }
    /// Getter
    pub fn mutate_only_prob(&self) -> &f64 {
        &self.mutate_only_prob
    }
}



// /// ///////////////////////////////////////////
// /// The NEAT Parameters class
// /// ///////////////////////////////////////////
// struct Parameters {
//     // Size of population
//     pub population_size: u32,
//     // If true, this enables dynamic compatibility thresholding
//     // It will keep the number of species between min_species and max_species
//     pub dynamic_compatibility: bool,
//     // Minimum number of species
//     pub min_species: u32,
//     // Maximum number of species
//     pub max_species: u32,
//     // Don't wipe the innovation database each generation?
//     pub innovations_forever: bool,
//     // Allow clones or nearly identical genomes to exist simultaneously in the population.
//     // This is useful for non-deterministic environments,
//     // as the same individual will get more than one chance to prove himself, also
//     // there will be more chances the same individual to mutate in different ways.
//     // The drawback is greatly increased time for reproduction. If you want to
//     // search quickly, yet less efficient, leave this to true.
//     pub allow_clones: bool,
//     // GA Parameters
//     //
//     // Age treshold, meaning if a species is below it, it is considered young
//     pub young_age_treshold: u32,
//     // Fitness boost multiplier for young species (1.0 means no boost)
//     // Make sure it is >= 1.0 to avoid confusion
//     pub young_age_fitness_boost: f64,
//     // Number of generations without improvement (stagnation) allowed for a species
//     pub species_max_stagnation: u32,
//     // Minimum jump in fitness necessary to be considered as improvement.
//     // Setting this value to 0.0 makes the system to behave like regular NEAT.
//     pub stagnation_delta: f64,
//     // Age threshold, meaning if a species if above it, it is considered old
//     pub old_age_treshold: u32,
//     // Multiplier that penalizes old species.
//     // Make sure it is < 1.0 to avoid confusion.
//     pub old_age_penalty: f64,
//     // Detect competetive coevolution stagnation
//     // This kills the worst species of age >N (each X generations)
//     pub detect_competetive_coevolution_stagnation: bool,
//     // Each X generation..
//     pub kill_worst_species_each: u32,
//     // Of age above..
//     pub kill_worst_age: u32,
//     // Percent of best individuals that are allowed to reproduce. 1.0 = 100%
//     pub survival_rate: f64,
//     // Probability for a baby to result from sexual reproduction (crossover/mating). 1.0 = 100%
//     pub crossover_rate: f64,
//     // If a baby results from sexual reproduction, this probability determines if mutation will
//     // be performed after crossover. 1.0 = 100% (always mutate after crossover)
//     pub overall_mutation_rate: f64,
//     // Probability for a baby to result from inter-species mating.
//     pub interspecies_crossover_rate: f64,
//     // Probability for a baby to result from Multipoint Crossover when mating. 1.0 = 100%
//     // The default if the Average mating.
//     pub multipoint_crossover_rate: f64,
//     // Performing roulette wheel selection or not?
//     pub roulette_wheel_selection: bool,
//     // For tournament selection
//     pub tournament_size: u32,
//     // Fraction of individuals to be copied unchanged
//     pub elite_fraction: f64,
//
//     // Phased Search parameters   //
//     //
//     // Using phased search or not
//     pub phased_searching: bool,
//     // Using delta coding or not
//     pub delta_coding: bool,
//     // What is the MPC + base MPC needed to begin simplifying phase
//     pub simplifying_phase_mpctreshold: u32,
//     // How many generations of global stagnation should have passed to enter simplifying phase
//     pub simplifying_phase_stagnation_treshold: u32,
//     // How many generations of MPC stagnation are needed to turn back on complexifying
//     pub complexity_floor_generations: u32,
//
//     // Novelty Search parameters       //
//     //
//     // the K constant
//     pub novelty_search_k: u32,
//     // Sparseness treshold. Add to the archive if above
//     pub novelty_search_p_min: f64,
//     // Dynamic Pmin?
//     pub novelty_search_dynamic_pmin: bool,
//     // How many evaluations should pass without adding to the archive
//     // in order to lower Pmin
//     pub novelty_search_no_archiving_stagnation_treshold: u32,
//     // How should it be multiplied (make it less than 1.0)
//     pub novelty_search_pmin_lowering_multiplier: f64,
//     // Not lower than this value
//     pub novelty_search_pmin_min: f64,
//     // How many one-after-another additions to the archive should
//     // pass in order to raise Pmin
//     pub novelty_search_quick_archiving_min_evaluations: u32,
//     // How should it be multiplied (make it more than 1.0)
//     pub novelty_search_pmin_raising_multiplier: f64,
//     // Per how many evaluations to recompute the sparseness
//     pub novelty_search_recompute_sparseness_each: u32,
//
//     // Mutation parameters
//     //
//     // Probability for a baby to be mutated with the Add-Neuron mutation.
//     pub mutate_add_neuron_prob: f64,
//     // Allow splitting of any recurrent links
//     pub split_recurrent: bool,
//     // Allow splitting of looped recurrent links
//     pub split_looped_recurrent: bool,
//     // Maximum number of tries to find a link to split
//     pub neuron_tries: u32,
//     // Probability for a baby to be mutated with the Add-Link mutation
//     pub mutate_add_link_prob: f64,
//     // Probability for a new incoming link to be from the bias neuron: f64,
//     pub mutate_add_link_from_bias_prob: f64,
//     // Probability for a baby to be mutated with the Remove-Link mutation
//     pub mutate_rem_link_prob: f64,
//     // Probability for a baby that a simple neuron will be replaced with a link
//     pub mutate_rem_simple_neuron_prob: f64,
//     // Maximum number of tries to find 2 neurons to add/remove a link
//     pub link_tries: u32,
//     // Probability that a link mutation will be made recurrent
//     pub recurrent_prob: f64,
//     // Probability that a recurrent link mutation will be looped
//     pub recurrent_loop_prob: f64,
//     // Probability for a baby's weights to be mutated
//     pub mutate_weights_prob: f64,
//     // Probability for a severe (shaking) weight mutation
//     pub mutate_weights_severe_prob: f64,
//     // Probability for a particular gene to be mutated. 1.0 = 100%
//     pub weight_mutation_rate: f64,
//     // Maximum perturbation for a weight mutation
//     pub weight_mutation_max_power: f64,
//     // Maximum magnitude of a replaced weight
//     pub weight_replacement_max_power: f64,
//     // Maximum absolute magnitude of a weight
//     pub max_weight: f64,
//     // Probability for a baby's A activation function parameters to be perturbed
//     pub mutate_activation_aprob: f64,
//     // Probability for a baby's B activation function parameters to be perturbed
//     pub mutate_activation_bprob: f64,
//     // Maximum magnitude for the A parameter perturbation
//     pub activation_amutation_max_power: f64,
//     // Maximum magnitude for the B parameter perturbation
//     pub activation_bmutation_max_power: f64,
//     // Maximum magnitude for time costants perturbation
//     pub time_constant_mutation_max_power: f64,
//     // Maximum magnitude for biases perturbation
//     pub bias_mutation_max_power: f64,
//     // Activation parameter A min/max
//     pub min_activation_a: f64,
//     pub max_activation_a: f64,
//     // Activation parameter B min/max
//     pub min_activation_b: f64,
//     pub max_activation_b: f64,
//     // Probability for a baby that an activation function type will be changed for a single neuron
//     // considered a structural mutation because of the large impact on fitness
//     pub mutate_neuron_activation_type_prob: f64,
//     // Probabilities for a particular activation function appearance
//     pub activation_function_signed_sigmoid_prob: f64,
//     pub activation_function_unsigned_sigmoid_prob: f64,
//     pub activation_function_tanh_prob: f64,
//     pub activation_function_tanh_cubic_prob: f64,
//     pub activation_function_signed_step_prob: f64,
//     pub activation_function_unsigned_step_prob: f64,
//     pub activation_function_signed_gauss_prob: f64,
//     pub activation_function_unsigned_gauss_prob: f64,
//     pub activation_function_abs_prob: f64,
//     pub activation_function_signed_sine_prob: f64,
//     pub activation_function_unsigned_sine_prob: f64,
//     pub activation_function_linear_prob: f64,
//     pub activation_function_relu_prob: f64,
//     pub activation_function_softplus_prob: f64,
//     // Probability for a baby's neuron time constant values to be mutated
//     pub mutate_neuron_time_constants_prob: f64,
//     // Probability for a baby's neuron bias values to be mutated
//     pub mutate_neuron_biases_prob: f64,
//     // Time constant range
//     pub min_neuron_time_constant: f64,
//     pub max_neuron_time_constant: f64,
//     // Bias range
//     pub min_neuron_bias: f64,
//     pub max_neuron_bias: f64,
//
//     // Speciation parameters
//     //
//     // Percent of disjoint genes importance
//     pub disjoint_coeff: f64,
//     // Percent of excess genes importance
//     pub excess_coeff: f64,
//     // Node-specific activation parameter A difference importance
//     pub activation_adiff_coeff: f64,
//     // Node-specific activation parameter B difference importance
//     pub activation_bdiff_coeff: f64,
//     // Average weight difference importance
//     pub weight_diff_coeff: f64,
//     // Average time constant difference importance
//     pub time_constant_diff_coeff: f64,
//     // Average bias difference importance
//     pub bias_diff_coeff: f64,
//     // Activation function type difference importance
//     pub activation_function_diff_coeff: f64,
//     // Compatibility treshold
//     pub compat_treshold: f64,
//     // Minumal value of the compatibility treshold
//     pub min_compat_treshold: f64,
//     // Modifier per generation for keeping the species stable
//     pub compat_treshold_modifier: f64,
//     // Per how many generations to change the treshold
//     pub compat_tresh_change_interval_generations: u32,
//     // Per how many evaluations to change the treshold
//     pub compat_tresh_change_interval_evaluations: u32,
//
//     // ES hyper_nEAT params
//     pub division_threshold: f64,
//     pub variance_threshold: f64,
//     // Used for Band prunning.
//     pub band_threshold: f64,
//     // Max and Min Depths of the quadtree
//     pub initial_depth: u32,
//     pub max_depth: u32,
//     // How many hidden layers before connecting nodes to output. At 0 there is
//     // one hidden layer. At 1, there are two and so on.
//     pub iteration_level: u32,
//     // The Bias value for the CPPN queries.
//     pub cppn_bias: f64,
//     // Quadtree Dimensions
//     // The range of the tree. Typically set to 2,
//     pub width: f64,
//     pub height: f64,
//     // The (x, y) coordinates of the tree
//     pub qtree_x: f64,
//     pub qtree_y: f64,
//     // Use Link Expression output
//     pub leo: bool,
//     // Threshold above which a connection is expressed
//     pub leo_threshold: f64,
//     // Use geometric seeding. Currently only along the X axis. 1
//     pub leo_seed: bool,
//     pub geometry_seed: bool,
// }
//
// impl Parameters {
//     pub fn new() -> Parameters {
//         Parameters {
//             population_size: 300,
//             dynamic_compatibility: true,
//             min_species: 5,
//             max_species: 10,
//             innovations_forever: true,
//             allow_clones: true,
//             young_age_treshold: 5,
//             young_age_fitness_boost: 1.1,
//             species_max_stagnation: 50,
//             stagnation_delta: 0.0,
//             old_age_treshold: 30,
//             old_age_penalty: 1.0,
//             detect_competetive_coevolution_stagnation: false,
//             kill_worst_species_each: 15,
//             kill_worst_age: 10,
//             survival_rate: 0.25,
//             crossover_rate: 0.7,
//             overall_mutation_rate: 0.25,
//             interspecies_crossover_rate: 0.0001,
//             multipoint_crossover_rate: 0.75,
//             roulette_wheel_selection: false,
//             tournament_size: 4,
//             elite_fraction: 0.01,
//             phased_searching: false,
//             delta_coding: false,
//             simplifying_phase_mpctreshold: 20,
//             simplifying_phase_stagnation_treshold: 30,
//             complexity_floor_generations: 40,
//             novelty_search_k: 15,
//             novelty_search_p_min: 0.5,
//             novelty_search_dynamic_pmin: true,
//             novelty_search_no_archiving_stagnation_treshold: 150,
//             novelty_search_pmin_lowering_multiplier: 0.9,
//             novelty_search_pmin_min: 0.05,
//             novelty_search_quick_archiving_min_evaluations: 8,
//             novelty_search_pmin_raising_multiplier: 1.1,
//             novelty_search_recompute_sparseness_each: 25,
//             mutate_add_neuron_prob: 0.01,
//             split_recurrent: true,
//             split_looped_recurrent: true,
//             neuron_tries: 6,
//             mutate_add_link_prob: 0.03,
//             mutate_add_link_from_bias_prob: 0.0,
//             mutate_rem_link_prob: 0.0,
//             mutate_rem_simple_neuron_prob: 0.0,
//             link_tries: 32,
//             recurrent_prob: 0.25,
//             recurrent_loop_prob: 0.25,
//             mutate_weights_prob: 0.90,
//             mutate_weights_severe_prob: 0.25,
//             weight_mutation_rate: 1.0,
//             weight_replacement_max_power: 1.0,
//             max_weight: 8.0,
//             mutate_activation_aprob: 0.0,
//             mutate_activation_bprob: 0.0,
//             activation_amutation_max_power: 0.0,
//             activation_bmutation_max_power: 0.0,
//             min_activation_a: 1.0,
//             max_activation_a: 1.0,
//             min_activation_b: 0.0,
//             max_activation_b: 0.0,
//             time_constant_mutation_max_power: 0.0,
//             bias_mutation_max_power: 1.0,
//             weight_mutation_max_power: 1.0,
//             mutate_neuron_time_constants_prob: 0.0,
//             mutate_neuron_biases_prob: 0.0,
//             min_neuron_time_constant: 0.0,
//             max_neuron_time_constant: 0.0,
//             min_neuron_bias: 0.0,
//             max_neuron_bias: 0.0,
//             mutate_neuron_activation_type_prob: 0.0,
//             activation_function_signed_sigmoid_prob: 0.0,
//             activation_function_unsigned_sigmoid_prob: 1.0,
//             activation_function_tanh_prob: 0.0,
//             activation_function_tanh_cubic_prob: 0.0,
//             activation_function_signed_step_prob: 0.0,
//             activation_function_unsigned_step_prob: 0.0,
//             activation_function_signed_gauss_prob: 0.0,
//             activation_function_unsigned_gauss_prob: 0.0,
//             activation_function_abs_prob: 0.0,
//             activation_function_signed_sine_prob: 0.0,
//             activation_function_unsigned_sine_prob: 0.0,
//             activation_function_linear_prob: 0.0,
//             activation_function_relu_prob: 0.0,
//             activation_function_softplus_prob: 0.0,
//             disjoint_coeff: 1.0,
//             excess_coeff: 1.0,
//             weight_diff_coeff: 0.5,
//             activation_adiff_coeff: 0.0,
//             activation_bdiff_coeff: 0.0,
//             time_constant_diff_coeff: 0.0,
//             bias_diff_coeff: 0.0,
//             activation_function_diff_coeff: 0.0,
//             compat_treshold: 5.0,
//             min_compat_treshold: 0.2,
//             compat_treshold_modifier: 0.3,
//             compat_tresh_change_interval_generations: 1,
//             compat_tresh_change_interval_evaluations: 10,
//             division_threshold: 0.03,
//             variance_threshold: 0.03,
//             band_threshold: 0.3,
//             initial_depth: 3,
//             max_depth: 3,
//             iteration_level: 1,
//             cppn_bias: 1.0,
//             width: 2.0,
//             height: 2.0,
//             qtree_x: 0.0,
//             qtree_y: 0.0,
//             leo: false,
//             leo_threshold: 0.1,
//             leo_seed: false,
//             geometry_seed: false,
//         }
//     }
// }
