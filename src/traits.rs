use rand;
/// TRAIT: A Trait is a group of parameters that can be expressed
///        as a group more than one time.  Traits save a genetic
///        algorithm from having to search vast parameter landscapes
///        on every node.  Instead, each node can simply point to a trait
///        and those traits can evolve on their own
#[derive(PartialEq, PartialOrd, Default, Clone)]
pub struct Traits {
    // ************ LEARNING PARAMETERS ***********
    // The following parameters are for use in
    //   neurons that learn through habituation,
    //   sensitization, or Hebbian-type processes
    trait_id: u32, // Used in file saving and loading
    params: Vec<f64>, // Keep traits in an array
}

impl Traits {
    /// construct new Traits
    pub fn new(id: u32, params: Vec<f64>) -> Traits {
        Traits {
            trait_id: id,
            params: params,
        }
    }
    /// Getter
    pub fn trait_id(&self) -> u32 {
        self.trait_id
    }
    /// Mutate this trait
    pub fn mutate(&mut self, probability: f64, mutation_power: f64) {
        self.params.iter_mut().map(|&mut x| {
            let float = rand::random::<f64>();
            if float > probability {
                let posneg = if rand::random::<bool>() { -1f64 } else { 1f64 };
                posneg * float * mutation_power
            } else {
                x
            }
        });
    }
    /// Create this Traits from two traits passed in
    /// id will be the id of trait1
    pub fn from_existing(trait1: Traits, trait2: Traits) -> Traits {
        Traits {
            trait_id: trait1.trait_id,
            params: trait1.params
                .iter()
                .zip(trait2.params.iter())
                .map(|x| *x.0 + *x.1 / 2.0f64)
                .collect::<Vec<f64>>(),
        }

    }
}
