use link_gene::LinkGene;
use neuron_gene::NeuronGene;

#[derive(PartialEq, PartialOrd)]
pub enum InnovationType {
    // neuron_id, type
    Neuron(NeuronGene),
    // from, to
    Link(LinkGene),
}

/// This Innovation class serves as a way to record innovations
///   specifically, so that an innovation in one genome can be
///   compared with other innovations in the same epoch, and if they
///   are the same innovation, they can both be assigned the same
///   innovation number.
///
///  This class can encode innovations that represent a new link
///  forming, or a new node being added.  In each case, two
///  nodes fully specify the innovation and where it must have
///  occured.  (Between them)
#[derive(PartialEq, PartialOrd)]
pub struct Innovation {
    id: u64, // Also doubles as neuron id.
    innovation_type: InnovationType,
    innovation_num1: u64, // The number assigned to the innovation
    innovation_num2: u64, /* If this is a new node innovation, then there are 2 innovations (links)
                           * added for the new node */
}

// impl Innovation {
// /// Constructor for the new node case
// pub fn new(int nin,int nout,double num1,double num2,int newid,double
// oldinnov)-> Innovation {
// 	Innovation {
//         innovation_type:NEWNODE,
// 	node_in_id:nin,
// 	node_out_id:nout,
// 	innovation_num1:num1,
// 	innovation_num2:num2,
// 	newnode_id:newid,
// 	old_innov_num:oldinnov,
//
// 	//Unused parameters set to zero
// 	new_weight:0,
// 	new_traitnum:0,
// 	recur_flag:false,
//     }
// }
// /// Constructor for new link case
// Innovation::Innovation(int nin,int nout,double num1,double w,int t) {
// 	innovation_type=NEWLINK;
// 	node_in_id=nin;
// 	node_out_id=nout;
// 	innovation_num1=num1;
// 	new_weight=w;
// 	new_traitnum=t;
//
// 	//Unused parameters set to zero
// 	innovation_num2=0;
// 	newnode_id=0;
// 	recur_flag=false;
// }
// /// Constructor for a recur link
// Innovation::Innovation(int nin,int nout,double num1,double w,int t,bool
// recur) {
// 	innovation_type=NEWLINK;
// 	node_in_id=nin;
// 	node_out_id=nout;
// 	innovation_num1=num1;
// 	new_weight=w;
// 	new_traitnum=t;
//
// 	//Unused parameters set to zero
// 	innovation_num2=0;
// 	newnode_id=0;
// 	recur_flag=recur;
// }
//
// impl Innovation {
//     /// Construct a new innovation
//     pub fn new(id: u64, innov_type: InnovationType) -> Innovation {
//         Innovation {
//             id: id,
//             innovation_type: innov_type,
//         }
//     }
// }
//
// 		//Constructor for the new node case
// Innovation(int nin,int nout,double num1,double num2,int newid,double
// oldinnov);
//
// 		//Constructor for new link case
// 		Innovation(int nin,int nout,double num1,double w,int t);
//
// 		//Constructor for a recur link
// 		Innovation(int nin,int nout,double num1,double w,int t,bool recur);
