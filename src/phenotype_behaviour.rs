use genome::Genome;
use rulinalg::matrix::Matrix;

pub trait PhenoTypeBehaviour {
    fn aquire(&self, genome: &Genome) -> bool {}

    fn distance_to(&self, pehnotype_beaviour: &Matrix) -> f64 {}

    fn successful(&self) -> bool {}
}
