use link_gene::LinkGene;
use std::cmp::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use traits::Traits;

/// Neuron types
#[derive(PartialEq, PartialOrd)]
pub enum NeuronType {
    Neuron(Vec<LinkGene>), // imput links
    Sensor(f64), // output values
}

/// All possible neuron places
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum NeuronPlace {
    In,
    Out,
    Hidden,
    Bias,
    NotUsed,
}

/// Activation function types
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum ActivationFunction {
    SignedSigmoid, // Sigmoid function   (default) (blurred cutting plane)
    UnSignedSigmoid,
    Tanh,
    TanhCubic,
    SignedStep, // Treshold (0 or 1)  (cutting plane)
    UnSignedStep,
    SignedGauss, // Gaussian           (symmetry)
    UnSignedGauss,
    Abs, // Absolute value |x| (another symettry)
    SignedSine, // Sine wave          (smooth repetition)
    UnSignedSine,
    Linear, // Linear f(x)=x      (combining coordinate frames only)
    Relu, // Rectifiers
    Softplus,
}



/// Nodes have link genes, these are weighted connections
/// to each other. In Nodes only have out link genes and likewise Out nodes only
/// have In link genes
/// Hidden Nodes have both in and out link genes.
/// A NODE is either a NEURON or a SENSOR.
///   - If it's a sensor, it can be loaded with a value for output
/// - If it's a neuron, it has a list of its incoming input signals
/// Use an activation count to avoid flushing
pub struct NeuronGene {
    innovation: u64,
    activation_count: u32, // keeps track of which activation the node is currently in
    last_activation: f64, // Holds the previous step's activation for recurrency
    last_activation2: f64, // Holds the activation BEFORE the prevous step's
    // This is necessary for a special recurrent case when the innode
    // of a recurrent link is one time step ahead of the outnode.
    // The innode then needs to send from TWO time steps ago
    nodetrait: Traits, // Points to a trait of parameters
    trait_id: u32, // identify the trait derived by this node
    dup: Arc<Mutex<NeuronGene>>, // Used for Genome duplication
    analogue: Arc<Mutex<NeuronGene>>, // Used for Gene decoding
    overriden: bool, // The NNode cannot compute its own output- something is overriding it
    override_value: f64, // Contains the activation value that will override this node's activation
    // Pointer to the Sensor corresponding to this Body.
    // Sensor* mySensor;
    frozen: bool, // When frozen, cannot be mutated (meaning its trait pointer is fixed)
    ftype: ActivationFunction, // type is either SIGMOID ..or others that can be added
    nodetype: NeuronType, // type is either NEURON or SENSOR
    activesum: f64, // The incoming activity before being processed
    activation: f64, // The total activation entering the NNode
    active_flag: bool, // To make sure outputs are active
    // NOT USED IN NEAT - covered by "activation" above
    output: f64, // Output of the NNode- the value in the NNode
    // ************ LEARNING PARAMETERS ***********
    // The following parameters are for use in
    //   neurons that learn through habituation,
    //   sensitization, or Hebbian-type processes
    params: Vec<f64>,
    incoming: Vec<Arc<Mutex<LinkGene>>>, /* A list of pointers to incoming weighted signals from other
                                          * nodes */
    outgoing: Vec<Arc<Mutex<LinkGene>>>, // A list of pointers to links carrying this node's signal
    // ###################################
    // These members are used for graphing
    // ###################################
    rowlevels: Vec<f64>, // Depths from output where this node appears
    row: u32, // Final row decided upon for drawing this NNode in
    ypos: u32,
    xpos: u32,
    node_id: u32, // A node can be given an identification number for saving in files
    gen_node_label: NeuronPlace, // Used for genetic marking of nodes
}

impl NeuronGene {
    // 		NNode(nodetype ntype,int nodeid);
    //
    // 		NNode(nodetype ntype,int nodeid, nodeplace placement);
    //
    // 		// Construct a NNode off another NNode for genome purposes
    // 		NNode(NNode *n,Trait *t);
    //
    // // Construct the node out of a file specification using given list of
    // traits
    // 		NNode (const char *argline, std::vector<Trait*> &traits);
    //
    // 		// Copy Constructor
    // 		NNode (const NNode& nnode);
    //
    // 		~NNode();
    //
    // 		// Just return activation for step
    // 		double get_active_out();
    //
    // 		// Return activation from PREVIOUS time step
    // 		double get_active_out_td();
    //
    // Returns the type of the node, NEURON or SENSOR
    pub fn get_type(&self) -> &NeuronType {
        &self.nodetype
    }
    //
    /// Allows alteration between NEURON and SENSOR.  Returns its argument
    pub fn set_type(&mut self, nodetype: NeuronType) {
        self.nodetype = nodetype;
    }

    /// Getter
    pub fn innovation(&self) -> u64 {
        self.innovation
    }

    /// If the node is a SENSOR, returns true and loads the value
    pub fn sensor_load(&mut self, load: f64) -> bool {
        match self.nodetype {
            NeuronType::Sensor(_) => {
                self.last_activation2 = self.last_activation;
                self.last_activation = self.activation;
                self.activation_count += 1;
                self.activation = load;
                true
            }
            NeuronType::Neuron(_) => false,
        }
    }
    /// Note: NEAT keeps track of which links are recurrent and which
    /// are not even though this is unnecessary for activation.
    /// It is useful to do so for 2 other reasons:
    /// 1. It makes networks visualization of recurrent networks possible
    /// 2. It allows genetic control of the proportion of connections
    ///    that may become recurrent
    /// Add an incoming connection a node
    /// Adds a NONRECURRENT Link to a new NNode in the incoming List
    pub fn add_incoming_connection(node: &NeuronGene, weight: f64) {}

    /// Adds a RECURRENT Link to a new NNode in the incoming List
    pub fn add_incoming_recurrent_conneciton(node: &NeuronGene, weight: f64) {}
    // 		// Recursively deactivate backwards through the network
    // 		void flushback();
    //
    // 		// Verify flushing for debugging
    // 		void flushback_check(std::vector<NNode*> &seenlist);
    //
    // 		// Print the node to a file
    //         void  print_to_file(std::ostream &outFile);
    // 	void print_to_file(std::ofstream &outFile);
    //
    // 		// Have NNode gain its properties from the trait
    // 		void derive_trait(Trait *curtrait);
    //
    // 		// Returns the gene that created the node
    // 		NNode *get_analogue();
    //
    // 		// Force an output value on the node
    // 		void override_output(double new_output);
    //
    // 		// Tell whether node has been overridden
    // 		bool overridden();
    //
    // 		// Set activation to the override value and turn off override
    // 		void activate_override();
    //
    // 		// Writes back changes weight values into the genome
    // 		// (Lamarckian trasnfer of characteristics)
    // 		void Lamarck();
    //
    // 		//Find the greatest depth starting from this neuron at depth d
    // 		int depth(int d,Network *mynet,int& count, int thresh);
    // }
}

impl Eq for NeuronGene {}

impl PartialEq for NeuronGene {
    fn eq(&self, other: &NeuronGene) -> bool {
        self.innovation == other.innovation
    }
}


impl PartialOrd for NeuronGene {
    fn partial_cmp(&self, other: &NeuronGene) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NeuronGene {
    fn cmp(&self, other: &NeuronGene) -> Ordering {
        self.innovation.cmp(&other.innovation)
    }
}
mod tests {}
