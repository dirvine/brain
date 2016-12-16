use genome::Genome;
use network::Network;

const ARCHIVE_SEED_AMOUNT: usize = 1;


// a novelty item is a "stake in the ground" i.e. a novel phenotype
#[derive(Clone)]
pub struct NoveltyItem {
    added: bool,
    indiv_number: u32,
    // we can keep track of genotype & phenotype of novel item
    genotype: Option<Genome>,
    phenotype: Option<Network>,

    // used to collect data
    data: Vec<Vec<f64>>,

    // future use
    age: f64,

    // used for analysis purposes
    novelty: f64,
    fitness: f64,
    generation: f64,
}

impl NoveltyItem {
    /// Initialise
    pub fn new() -> NoveltyItem {
        NoveltyItem {
            added: false,
            genotype: None,
            phenotype: None,
            age: 0.0,
            generation: 0.0,
            indiv_number: 0,
            data: Vec::new(),
            fitness: 0f64,
            novelty: 0f64,
        }
    }

    // TODO impl Display for NoveltyItem
    // pub fn print(&self) {
    //     let point =
    //     for i in &self.data {
    //         for j in &i {
    //             j
    //         }
    //     }
    //     println!("Novelty : {}, Fitenss : {}, Generation {}, Indiv {}, Point : {}", );
    // }
}
