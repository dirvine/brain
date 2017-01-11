#![deny(missing_docs,
        trivial_casts, trivial_numeric_casts,
        unsafe_code,
        unused_import_braces, unused_qualifications)]

//! Implementation of NeuroEvolution of Augmenting Topologies [NEAT]
//! (http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
//! This implementation is a playground for the moment.

#[macro_use]
extern crate log;
extern crate rand;
extern crate rulinalg;
extern crate num_cpus;
extern crate crossbeam;


mod genome;
/// A gene holds a `Link`. It alse records the mutation quantity of the `Link` and whether it is
/// frozen (no longer able to be mutated) or enabled/disabled
pub mod gene;
/// A link connects 2 `NetworkNodes` and has a `Weight` associated with it.
pub mod link;
/// A `NetworkNode` can be an Input (Sensor), Output or Hidden.
pub mod network_node;
mod innovation_database;
mod innovation;
mod organism;
mod species;
mod population;
mod parameters;
mod selection;
mod traits;
mod network;
mod novelty_item;
mod weight;


use rand::distributions::{IndependentSample, Range};
use rand::distributions::normal::StandardNormal;

fn rand_float() -> f64 {
    let mut rng = rand::thread_rng();
    Range::new(-1f64, 1.).ind_sample(&mut rng)
}

fn gaussrand() -> f64 {
    let StandardNormal(x) = rand::random();
    x
}

fn sigmoid(activesum: f64, slope: f64) -> f64 {
    1.0 / (1.0 + (-(slope * activesum).exp())) //Compressed
}

fn hebbian(weight: f64,
           maxweight: f64,
           active_in: f64,
           active_out: f64,
           hebb_rate: f64,
           pre_rate: f64,
           post_rate: f64)
           -> f64 {

    let mut neg = false;
    let mut delta: f64;

    let mweight = if maxweight < 5.0 { 5.0 } else { maxweight };

    let w = if weight > maxweight {
        maxweight
    } else {
        weight
    };

    if w < 0.0 {
        neg = true;
    }


    let mut topweight = weight + 2.0;
    if topweight > maxweight {
        topweight = maxweight
    };

    if !neg {
        delta = hebb_rate * (maxweight - weight) * active_in * active_out +
                pre_rate * (topweight) * active_in * (active_out - 1.0);

        return weight + delta;

    } else {
        // In the inhibatory case, we strengthen the synapse when output is low and
        // input is high
        delta = pre_rate*(maxweight-weight)*active_in*(1.0-active_out)+ //"unhebb"
			-hebb_rate*(topweight+2.0)*active_in*active_out; //anti-hebbian

        return -(weight + delta).abs();

    }

}

// Cerebrum -> Large superior region of brain
// consists of
// frontal lobe
// temporal lobe
// pariental lobe
// occipital lobe
//
// Function of some of the sub-divisions / regions
//
// Movement
// Sensory processing (visual hearing smell touch taste)
// Olfacation - smell (large part of cortex)
// Language and communication
// Learning and memory
//
// inputs are nerves
//
// Neuron includes Genomes
//
// axon links neurons and synapses connect them (firing signals)
//
// axons also link lobes (cortex) (species) together.
