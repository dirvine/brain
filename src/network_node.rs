use link::Link;
use std::cmp::Ordering;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use traits::Traits;

/// Node types
#[derive(PartialEq, PartialOrd, Clone)]
pub enum NodeType {
    Sensor,
    Input,
}


/// Activation function types
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone)]
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


/// Nodes have links these are weighted connections
/// to each other. In Nodes only have out links and likewise Out nodes only
/// have In links
/// Hidden Nodes have both in and out links.
/// Use an activation count to avoid flushing
#[derive(Clone)]
pub struct NetworkNode {
    innovation: u64,
    activation_count: u32, // keeps track of which activation the node is currently in
    last_activation: f64, // Holds the previous step's activation for recurrency
    last_activation2: f64, // Holds the activation BEFORE the prevous step's
    // This is necessary for a special recurrent case when the innode
    // of a recurrent link is one time step ahead of the outnode.
    // The innode then needs to send from TWO time steps ago
    nodetrait: Traits, // Points to a trait of parameters
    trait_id: u32, // identify the trait derived by this node
    dup: Arc<Mutex<NetworkNode>>, // Used for Genome duplication
    analogue: Arc<Mutex<NetworkNode>>, // Used for Gene decoding
    overriden: bool, // The NNode cannot compute its own output- something is overriding it
    override_value: f64, // Contains the activation value that will override this node's activation
    // Pointer to the Sensor corresponding to this Body.
    // Sensor* mySensor;
    frozen: bool, // When frozen, cannot be mutated (meaning its trait pointer is fixed)
    activation_function: ActivationFunction, // type is either SIGMOID ..or others that can be added
    nodetype: NodeType, // type is either NEURON or SENSOR
    activesum: f64, // The incoming activity before being processed
    activation: f64, // The total activation entering the NNode
    active_flag: bool, // To make sure outputs are active
    // ************ LEARNING PARAMETERS ***********
    // The following parameters are for use in
    //   neurons that learn through habituation,
    //   sensitization, or Hebbian-type processes
    params: Vec<f64>,
    incoming: Vec<Rc<Link>>, /* A list of pointers to incoming weighted signals from other
                              * nodes */
    outgoing: Vec<Rc<Link>>, // A list of pointers to links carrying this node's signal
}

impl NetworkNode {
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
    /// Getter
    pub fn neuron_type(&self) -> &NodeType {
        &self.nodetype
    }
    // /// Getter
    // pub fn gen_node_label(&self) -> &NodePlace {
    //     &self.gen_node_label
    // }
    //
    /// Allows alteration between NEURON and SENSOR.
    pub fn set_type(&mut self, nodetype: NodeType) {
        self.nodetype = nodetype;
    }

    /// Getter
    pub fn innovation(&self) -> u64 {
        self.innovation
    }

    /// Reset activations
    pub fn flush(&mut self) {
        self.activation_count = 0;
        self.last_activation = 0f64;
        self.last_activation2 = 0f64;
        self.activation = 0f64;
    }
    /// If the node is a SENSOR, returns true and loads the value
    pub fn sensor_load(&mut self, load: f64) -> bool {
        match self.nodetype {
            NodeType::Sensor => {
                self.last_activation2 = self.last_activation;
                self.last_activation = self.activation;
                self.activation_count += 1;
                self.activation = load;
                true
            }
            _ => false,
        }
    }
    /// Note: NEAT keeps track of which links are recurrent and which
    /// are not even though this is unnecessary for activation.
    /// It is useful to do so for 2 other reasons:
    /// 1. It makes networks visualization of recurrent networks possible
    /// 2. It allows genetic control of the proportion of connections
    ///    that may become recurrent
    /// Add an incoming connection a node
    pub fn add_incoming_connection(&mut self, node: Rc<Link>) {
        self.incoming.push(node);
    }
    /// Add an outgoing connection
    pub fn add_outgoing_connection(&mut self, node: Rc<Link>) {
        self.outgoing.push(node);
    }
    /// Add a recurrent connection
    pub fn add_recurrent_connection(&mut self, node: Rc<Link>) {
        unimplemented!()
        // self.outgoing.push(node);
        // self.incoming.push(node);
    }
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
}

impl Eq for NetworkNode {}

impl PartialEq for NetworkNode {
    fn eq(&self, other: &NetworkNode) -> bool {
        self.innovation == other.innovation
    }
}


impl PartialOrd for NetworkNode {
    fn partial_cmp(&self, other: &NetworkNode) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NetworkNode {
    fn cmp(&self, other: &NetworkNode) -> Ordering {
        self.innovation.cmp(&other.innovation)
    }
}
mod tests {}
