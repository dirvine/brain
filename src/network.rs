use genome::Genome;
use network_node::NodeType;

/// A NETWORK is a LIST of input NODEs and a LIST of output NODEs
///   The point of the network is to define a single entity which can evolve
///   or learn on its own, even though it may be part of a larger framework
#[derive(Clone, PartialEq)]
pub struct Network {
    genotype: Genome, // Allows Network to be matched with its Genome
    name: String, // Every Network or subNetwork can have a name
    net_id: u32, // Allow for a network id
    maxweight: f64, // Maximum weight in network for adaptation purposes
    adaptable: bool, // Tells whether network can adapt or not
}

impl Network {
    /// All nodes in network
    pub fn num_nodes(&self) -> usize {
        self.genotype.num_nodes()
    }
    /// All links in network
    pub fn links(&self) -> usize {
        unimplemented!()
    }
    /// All inout nodes
    pub fn num_inputs(&self) -> usize {
        self.genotype.num_inputs()
    }
    /// All output nodes
    pub fn num_outputs(&self) -> usize {
        self.genotype.num_outputs()
    }
    /// Puts the network back into an initial state
    fn flush(&mut self) {
        unimplemented!()
        // self.outputs().iter().map(|x| x.flushback());
    }

    /// If all output are not active then return true
    fn outputsoff(&mut self) -> bool {
        self.genotype.outputs_off()
    }

    /// Print the connections weights to a file separated by only carriage returns
    pub fn print_links(&self) {
        self.genotype.network_nodes.iter().map(|x| {
            // TODO
            if *x.borrow().node_type() != NodeType::Sensor {
                println!("In node Id: {:?} Out node Id {:?} Link weight {:?}",
                         x.borrow().node_id(),
                         x.borrow().node_id(),
                         24)
            }
        });

    }

    /// Activates the net such that all outputs are active
    /// Returns true on success;
    pub fn activate(&mut self) {
        println!("Activating network ");

        let mut abortcount = 0;  //Used in case the output is somehow truncated from the network

        // Keep activating until all the outputs have become active
        // This only happens on the first activation, because after that they
        // are always active)

        if self.outputsoff() {
            abortcount += 1;
            if abortcount == 20 {
                println!("Inputs disconnected from output!");
                return;
            }
            println!("Outputs are off");

            // For each node, compute the sum of its incoming activation
            for mut node in self.genotype
                .network_nodes
                .iter_mut()
                .filter(|x| *x.borrow().node_type() != NodeType::Sensor) {
                node.borrow_mut().add_active_sum(0.0);
                node.borrow_mut().set_active_flag(false);
                // For each incoming connection, add the activity from the connection to the activesum
                for mut link in node.borrow_mut().incoming_mut().iter() {
                    // Handle time delays
                    if !link.borrow_mut().time_delay() {
                        let add = link.borrow_mut().weight().0 *
                                  link.borrow().inode().borrow().activation();
                        if link.borrow().inode().borrow().active_flag() ||
                           *link.borrow().inode().borrow().node_type() == NodeType::Sensor {
                            node.borrow_mut().set_active_flag(true);
                            node.borrow_mut().add_active_sum(add);
                        }
                    } else {

                        let add = link.borrow().weight().0 *
                                  link.borrow().inode().borrow().active_out_time_delay();
                        node.borrow_mut().add_active_sum(add);
                    }

                }
            }

            // Now activate all the non-sensor nodes off their incoming activation
            for mut node in self.genotype.network_nodes.iter_mut().filter(|x| *x.borrow().node_type() != NodeType::Sensor && x.borrow().active_flag()) {
				//Only activate if some active input came in
					//"Activating "node_id" with "<<(*curnode)->activesum<<": ";

					//Keep a memory of activations for potential time delayed connections
					node.borrow_mut().set_penultimate_activation(node.borrow().last_activation());
					node.borrow_mut().set_last_activation(node.borrow().activation());

					//If the node is being overrided from outside,
					//stick in the override value
					if node.borrow().overridden() {
						//Set activation to the override value and turn off override
						node.borrow_mut().activate_override();
					}
					else {
						//Now run the net activation through an activation function
                        node.borrow_mut().set_activation(super::sigmoid(node.borrow().activation(), 4.924273));
					}
					//cout<<(*curnode)->activation<<endl;

					//Increment the activation_count
					//First activation cannot be from nothing!!
					node.borrow_mut().increment_activation_count();
		}

        }

        if self.adaptable {
            println!("ADAPTING");
            // ADAPTATION:  Adapt weights based on activations
            for mut node in self.genotype
                .network_nodes
                .iter()
                .filter(|x| *x.borrow().node_type() != NodeType::Sensor) {
                // cout<<"On node "<<(*curnode)->node_id<<endl;

                // For each incoming connection, perform adaptation based on the trait of the connection
                for mut link in node.borrow_mut().incoming().iter() {
                    if link.borrow().trait_id() == 2 || link.borrow().trait_id() == 3 ||
                       link.borrow().trait_id() == 4 {

                        // In the recurrent case we must take the last activation of the input for calculating hebbian changes
                        if link.borrow().recur() {
                            link.borrow_mut()
                                .set_weight(super::hebbian(link.borrow().weight().0,
                                                           self.maxweight,
                                                           link.borrow()
                                                               .inode()
                                                               .borrow()
                                                               .last_activation(),
                                                           link.borrow()
                                                               .onode()
                                                               .borrow()
                                                               .activation(), // FIXME active_out
                                                           link.borrow_mut().trait_params()[0],
                                                           link.borrow_mut().trait_params()[1],
                                                           link.borrow_mut().trait_params()[2]));


                        } else {
                            // non-recurrent case
                            link.borrow_mut()
                                .set_weight(super::hebbian(link.borrow().weight().0,
                                                           self.maxweight,
                                                           link.borrow()
                                                               .inode()
                                                               .borrow()
                                                               .activation(),
                                                           link.borrow()
                                                               .onode()
                                                               .borrow()
                                                               .activation(), // FIXE active_out
                                                           link.borrow_mut().trait_params()[0],
                                                           link.borrow_mut().trait_params()[1],
                                                           link.borrow_mut().trait_params()[2]));
                        }
                    }

                }


            }

        }

    }
}
// // Prints the values of its outputs
// void Network::show_activation() {
// 	std::vector<NNode*>::iterator curnode;
// 	int count;
//
// 	//if (name!=0)
// 	//  cout<<"Network "<<name<<" with id "<<net_id<<" outputs: (";
// 	//else cout<<"Network id "<<net_id<<" outputs: (";
//
// 	count=1;
// 	for(curnode=outputs.begin();curnode!=outputs.end();++curnode) {
// 		//cout<<"[Output #"<<count<<": "<<(*curnode)<<"] ";
// 		count++;
// 	}
//
// 	//cout<<")"<<endl;
// }
//
// void Network::show_input() {
// 	std::vector<NNode*>::iterator curnode;
// 	int count;
//
// 	//if (name!=0)
// 	//  cout<<"Network "<<name<<" with id "<<net_id<<" inputs: (";
// 	//else cout<<"Network id "<<net_id<<" outputs: (";
//
// 	count=1;
// 	for(curnode=inputs.begin();curnode!=inputs.end();++curnode) {
// 		//cout<<"[Input #"<<count<<": "<<(*curnode)<<"] ";
// 		count++;
// 	}
//
// 	//cout<<")"<<endl;
// }
//
// // Add an input
// void Network::add_input(NNode *in_node) {
// 	inputs.push_back(in_node);
// }
//
// // Add an output
// void Network::add_output(NNode *out_node) {
// 	outputs.push_back(out_node);
// }
//
// // Takes an array of sensor values and loads it into SENSOR inputs ONLY
// void Network::load_sensors(double *sensvals) {
// 	//int counter=0;  //counter to move through array
// 	std::vector<NNode*>::iterator sensPtr;
//
// 	for(sensPtr=inputs.begin();sensPtr!=inputs.end();++sensPtr) {
// 		//only load values into SENSORS (not BIASes)
// 		if (((*sensPtr)->type)==SENSOR) {
// 			(*sensPtr)->sensor_load(*sensvals);
// 			sensvals++;
// 		}
// 	}
// }
//
// void Network::load_sensors(const std::vector<float> &sensvals) {
// 	//int counter=0;  //counter to move through array
// 	std::vector<NNode*>::iterator sensPtr;
// 	std::vector<float>::const_iterator valPtr;
//
// 	for(valPtr = sensvals.begin(), sensPtr = inputs.begin(); sensPtr != inputs.end() && valPtr != sensvals.end(); ++sensPtr, ++valPtr) {
// 		//only load values into SENSORS (not BIASes)
// 		if (((*sensPtr)->type)==SENSOR) {
// 			(*sensPtr)->sensor_load(*valPtr);
// 			//sensvals++;
// 		}
// 	}
// }
//
//
// // Takes and array of output activations and OVERRIDES
// // the outputs' actual activations with these values (for adaptation)
// void Network::override_outputs(double* outvals) {
//
// 	std::vector<NNode*>::iterator outPtr;
//
// 	for(outPtr=outputs.begin();outPtr!=outputs.end();++outPtr) {
// 		(*outPtr)->override_output(*outvals);
// 		outvals++;
// 	}
//
// }
//
// void Network::give_name(char *newname) {
// 	char *temp;
// 	char *temp2;
// 	temp=new char[strlen(newname)+1];
// 	strcpy(temp,newname);
// 	if (name==0) name=temp;
// 	else {
// 		temp2=name;
// 		delete temp2;
// 		name=temp;
// 	}
// }
//
// // The following two methods recurse through a network from outputs
// // down in order to count the number of nodes and links in the network.
// // This can be useful for debugging genotype->phenotype spawning
// // (to make sure their counts correspond)
//
// int Network::nodecount() {
// 	int counter=0;
// 	std::vector<NNode*>::iterator curnode;
// 	std::vector<NNode*>::iterator location;
// 	std::vector<NNode*> seenlist;  //List of nodes not to doublecount
//
// 	for(curnode=outputs.begin();curnode!=outputs.end();++curnode) {
//
//         location = std::find(seenlist.begin(),seenlist.end(),(*curnode));
// 		if (location==seenlist.end()) {
// 			counter++;
// 			seenlist.push_back(*curnode);
// 			nodecounthelper((*curnode),counter,seenlist);
// 		}
// 	}
//
// 	numnodes=counter;
//
// 	return counter;
//
// }
//
// void Network::nodecounthelper(NNode *curnode,int &counter,std::vector<NNode*> &seenlist) {
// 	std::vector<Link*> innodes=curnode->incoming;
// 	std::vector<Link*>::iterator curlink;
// 	std::vector<NNode*>::iterator location;
//
// 	if (!((curnode->type)==SENSOR)) {
// 		for(curlink=innodes.begin();curlink!=innodes.end();++curlink) {
//             location= std::find(seenlist.begin(),seenlist.end(),((*curlink)->in_node));
// 			if (location==seenlist.end()) {
// 				counter++;
// 				seenlist.push_back((*curlink)->in_node);
// 				nodecounthelper((*curlink)->in_node,counter,seenlist);
// 			}
// 		}
//
// 	}
//
// }
//
// int Network::linkcount() {
// 	int counter=0;
// 	std::vector<NNode*>::iterator curnode;
// 	std::vector<NNode*> seenlist;  //List of nodes not to doublecount
//
// 	for(curnode=outputs.begin();curnode!=outputs.end();++curnode) {
// 		linkcounthelper((*curnode),counter,seenlist);
// 	}
//
// 	numlinks=counter;
//
// 	return counter;
//
// }
//
// void Network::linkcounthelper(NNode *curnode,int &counter,std::vector<NNode*> &seenlist) {
// 	std::vector<Link*> inlinks=curnode->incoming;
// 	std::vector<Link*>::iterator curlink;
// 	std::vector<NNode*>::iterator location;
//
//     location = std::find(seenlist.begin(),seenlist.end(),curnode);
// 	if ((!((curnode->type)==SENSOR))&&(location==seenlist.end())) {
// 		seenlist.push_back(curnode);
//
// 		for(curlink=inlinks.begin();curlink!=inlinks.end();++curlink) {
// 			counter++;
// 			linkcounthelper((*curlink)->in_node,counter,seenlist);
// 		}
//
// 	}
//
// }
//
// // Destroy will find every node in the network and subsequently
// // delete them one by one.  Since deleting a node deletes its incoming
// // links, all nodes and links associated with a network will be destructed
// // Note: Traits are parts of genomes and not networks, so they are not
// //       deleted here
// void Network::destroy() {
// 	std::vector<NNode*>::iterator curnode;
// 	std::vector<NNode*>::iterator location;
// 	std::vector<NNode*> seenlist;  //List of nodes not to doublecount
//
// 	// Erase all nodes from all_nodes list
//
// 	for(curnode=all_nodes.begin();curnode!=all_nodes.end();++curnode) {
// 		delete (*curnode);
// 	}
//
//
// 	// -----------------------------------
//
// 	//  OLD WAY-the old way collected the nodes together and then deleted them
//
// 	//for(curnode=outputs.begin();curnode!=outputs.end();++curnode) {
// 	//cout<<seenstd::vector<<endl;
// 	//cout<<curnode<<endl;
// 	//cout<<curnode->node_id<<endl;
//
// 	//  location=find(seenlist.begin(),seenlist.end(),(*curnode));
// 	//  if (location==seenlist.end()) {
// 	//    seenlist.push_back(*curnode);
// 	//    destroy_helper((*curnode),seenlist);
// 	//  }
// 	//}
//
// 	//Now destroy the seenlist, which is all the NNodes in the network
// 	//for(curnode=seenlist.begin();curnode!=seenlist.end();++curnode) {
// 	//  delete (*curnode);
// 	//}
// }
//
// void Network::destroy_helper(NNode *curnode,std::vector<NNode*> &seenlist) {
// 	std::vector<Link*> innodes=curnode->incoming;
// 	std::vector<Link*>::iterator curlink;
// 	std::vector<NNode*>::iterator location;
//
// 	if (!((curnode->type)==SENSOR)) {
// 		for(curlink=innodes.begin();curlink!=innodes.end();++curlink) {
//             location = std::find(seenlist.begin(),seenlist.end(),((*curlink)->in_node));
// 			if (location==seenlist.end()) {
// 				seenlist.push_back((*curlink)->in_node);
// 				destroy_helper((*curlink)->in_node,seenlist);
// 			}
// 		}
//
// 	}
//
// }
//
// // This checks a POTENTIAL link between a potential in_node and potential out_node to see if it must be recurrent
// bool Network::is_recur(NNode *potin_node,NNode *potout_node,int &count,int thresh) {
// 	std::vector<Link*>::iterator curlink;
//
// 	++count;  //Count the node as visited
//
// 	if (count>thresh) {
// 		//cout<<"returning false"<<endl;
// 		return true;  //Short out the whole thing- loop detected
// 	}
//
// 	if (potin_node==potout_node) return true;
// 	else {
// 		//Check back on all links...
// 		for(curlink=(potin_node->incoming).begin();curlink!=(potin_node->incoming).end();curlink++) {
// 			//But skip links that are already recurrent
// 			//(We want to check back through the forward flow of signals only
// 			if (!((*curlink)->is_recurrent)) {
// 				if (is_recur((*curlink)->in_node,potout_node,count,thresh)) return true;
// 			}
// 		}
// 		return false;
// 	}
// }
//
// bool Network::is_recur2(NNode *potin_node,NNode *potout_node,int &count, int thresh) {
// 	NNode* curnode;
// 	std::vector<NNode*> seenlist;  //List of nodes not to doublecount
// 	curnode=potin_node; //start at out node, if we find a path that lands at in_node, then this link will be recurrent
// 	seenlist.push_back(curnode);
// 	return is_rec_helper(curnode,potout_node,seenlist);
// }
//
// bool Network::is_rec_helper(NNode* curnode, NNode* find_node,std::vector<NNode*> &seenlist)
// {
// 	std::vector<Link*> innodes=curnode->incoming;
// 	std::vector<Link*>::iterator curlink;
// 	std::vector<NNode*>::iterator location;
// 	if (!((curnode->type)==SENSOR)) {
// 		for(curlink=innodes.begin();curlink!=innodes.end();++curlink) {
// 			if((*curlink)->in_node==find_node)
// 				return true;
//             location= std::find(seenlist.begin(),seenlist.end(),((*curlink)->in_node));
// 			if (location==seenlist.end()) {
// 				seenlist.push_back((*curlink)->in_node);
// 				if(is_rec_helper((*curlink)->in_node,find_node,seenlist))
// 					return true;
// 			}
// 		}
// 	}
// 	return false;
// }
//
// #<{(|
// int Network::nodecount() {
// 	int counter=0;
// 	std::vector<NNode*>::iterator curnode;
// 	std::vector<NNode*>::iterator location;
// 	std::vector<NNode*> seenlist;  //List of nodes not to doublecount
//
// 	for(curnode=outputs.begin();curnode!=outputs.end();++curnode) {
//
//         location = std::find(seenlist.begin(),seenlist.end(),(*curnode));
// 		if (location==seenlist.end()) {
// 			counter++;
// 			seenlist.push_back(*curnode);
// 			nodecounthelper((*curnode),counter,seenlist);
// 		}
// 	}
//
// 	numnodes=counter;
//
// 	return counter;
//
// }
//
// void Network::nodecounthelper(NNode *curnode,int &counter,std::vector<NNode*> &seenlist) {
// 	std::vector<Link*> innodes=curnode->incoming;
// 	std::vector<Link*>::iterator curlink;
// 	std::vector<NNode*>::iterator location;
//
// 	if (!((curnode->type)==SENSOR)) {
// 		for(curlink=innodes.begin();curlink!=innodes.end();++curlink) {
//             location= std::find(seenlist.begin(),seenlist.end(),((*curlink)->in_node));
// 			if (location==seenlist.end()) {
// 				counter++;
// 				seenlist.push_back((*curlink)->in_node);
// 				nodecounthelper((*curlink)->in_node,counter,seenlist);
// 			}
// 		}
//
// 	}
//
// }
// |)}>#
//
// int Network::input_start() {
// 	input_iter=inputs.begin();
// 	return 1;
// }
//
// int Network::load_in(double d) {
// 	(*input_iter)->sensor_load(d);
// 	input_iter++;
// 	if (input_iter==inputs.end()) return 0;
// 	else return 1;
// }
//
//
// //Find the maximum number of neurons between an ouput and an input
// int Network::max_depth() {
//   std::vector<NNode*>::iterator curoutput; //The current output we are looking at
//   int cur_depth; //The depth of the current node
//   int max=0; //The max depth
//   int count=0; //a way to prevent infinite loops
//   int thresh=(all_nodes.size())*(all_nodes.size());
//
//   for(curoutput=outputs.begin();curoutput!=outputs.end();curoutput++) {
//     cur_depth=(*curoutput)->depth(0,this,count,thresh);
//     if (cur_depth>max) max=cur_depth;
//   }
//
//   return max;
//
// }
// }
//
//
