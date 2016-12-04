#![deny(missing_docs,
        trivial_casts, trivial_numeric_casts,
        unsafe_code,
        unused_import_braces, unused_qualifications)]

#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]
#![cfg_attr(feature="clippy", deny(clippy, unicode_not_nfc, wrong_pub_self_convention))]
#![cfg_attr(feature="clippy", allow(use_debug, too_many_arguments))]

//! Implementation of NeuroEvolution of Augmenting Topologies [NEAT]
//! (http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
//! This implementation is a playground for the moment.

extern crate rand;
extern crate rulinalg;
extern crate num_cpus;
extern crate crossbeam;


mod neuron_gene;
mod genome;
mod link_gene;
mod innovation_database;
mod innovation;
mod parameters;
mod selection;


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
