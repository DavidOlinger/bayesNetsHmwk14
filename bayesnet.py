import pprint
import random
from typing import Optional

class BayesianNode:
    """A node in a Bayesian network, with discrete domain"""

    def __init__(self, name: str, values: list[str], parents: list[str],
                 cpt: dict[tuple[str, ...],dict[str,float]]):
        """ cpt key is tuple of (parent node 1's value, ..., parent node k's value)
            That yields a dictionary where the keys are this node's values,
            each of which maps to the probability of that value for this node, given its parents' values
        """
        self.name = name
        self.values = set(values)
        self.parents = parents

        self.__cpt = cpt

    def get_probability(self, node_value: str, parent_values: tuple[str, ...]) -> float:
        """ The probability of this node taking value node_value given the parents' values

            THe values in parent_values should match the ordering of parents in self.parents
           """

        if len(parent_values) != len(self.parents):
            raise ValueError("Number of parent values should match number of parents")
        
        if node_value not in self.values:
            raise ValueError(f"Invalid value {node_value} for node {self.name}")
        
        # Get the probability using the key
        try:
            prob = self.__cpt[parent_values][node_value]
        except KeyError:
            print(f"Error with key {(parent_values)},{node_value} for node {self.name} with parents {self.parents}")
            print(f"valid keys are: {self.__cpt.keys()}")
            return 0.0
        
        return prob
    

    def sample_value(self, parent_values: tuple[str, ...]) -> str:
        """ Randomly generate a value for this node, given its parent values """

        distn = self.__cpt[parent_values]

        # The [0] is because choices returns a list of length 1
        return random.choices(list(distn.keys()), list(distn.values()))[0]

    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        s = self.name + " in " + str(self.values) + "\n"
        s += "Parents: " + str(self.parents) + "\n"
        s += "CPT:" + "\n"
        s += pprint.pformat(self.__cpt, indent=4, sort_dicts=False)

        return s


def remove_duplicates_preserve_order(input_list):
    seen = set()
    new_list = []
    for item in input_list:
        if item not in seen:
            new_list.append(item)
            seen.add(item)
    return new_list


class BayesianNetwork:
    """ Represents a discrete-valued Bayesian network """

    def __init__(self, nodes: dict[str, BayesianNode], topo_order: list[str]):
        """ nodes maps variable names to the BayesianNode object
            topo_order has the same variable names, listed in topographical order (for easy inference)
        """
        self.nodes = nodes
        self.topo_order = topo_order

    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        s = ""
        
        for v in self.topo_order:
            s += str(self.nodes[v])
            s += "\n\n"
            
        return s
    
    def enumerate_variables_tuples(self, rvs: list[str]) -> list[tuple[str, ...]]:
        """Returns a list of all possible tuples for the rvs variables
        
        rvs is a list of variable names

        Example: rvs = ['A', 'Rainy', 'X'], with 'A' in ['1', '2', '3'], 'Rainy' in ['t', 'f'], 
            and 'X' in ['5', '10']. This function will return
            
            [('1', 't', '5'),
             ('1', 't', '10'),
             ('1', 'f', '5'),
             ('1', 'f', '10'),
             ('2', 't', '5'),
             ('2', 't', '10'),
             ('2', 'f', '5'),
             ('2', 'f', '10'),
             ('3', 't', '5'),
             ('3', 't', '10'),
             ('3', 'f', '5'),
             ('3', 'f', '10')]
        """

        head_vals = self.nodes[rvs[0]].values
        if len(rvs) == 1:
            return [(v,) for v in head_vals]
        
        tail_vals = self.enumerate_variables_tuples(rvs[1:])
        return [(v, *recursive_vals) for v in head_vals for recursive_vals in tail_vals]
    

    def get_parent_values(self, var_name: str, rv_values: dict[str,str]) -> tuple[str, ...]:
        """Extracts and returns the values of var_name's parents
        
            rv_values is a mapping of variable name to value;
            it should include values for all the parents of var_name,
            but may include other variables' values as well
             
            Raises KeyError if one of the parents of var_name is not in rv_values
        """
        return tuple([rv_values[p] for p in self.nodes[var_name].parents])


    def get_joint_prob(self, rv_values: dict[str, str]) -> float:
        """Get a joint probability of the given rv_values

            rv_values maps variable names to values

            For each variable x in the network, rv_values[x] should be a value in the domain of x.
            That is, all variables in the network should have a value specified in rv_values.
        """

        probability = 1.0
        for variable_name in self.topo_order:
            node = self.nodes[variable_name]
            parent_values = self.get_parent_values(variable_name, rv_values)
            probability *= node.get_probability(rv_values[variable_name], parent_values)
        return probability


    def get_marginal_prob(self, rv_values: dict[str, str]) -> float:
        """Compute the joint probability of the given rv_values

            rv_values maps variable names to values

            Unlike get_joint_prob, not all variables in the network need be specified in rv_values.
            For those that are, rv_values[x] should be a value in the domain of x
        """

        return self.__cond_prob_topo_order(rv_values, {}, 0)

                
    def get_conditional_prob(self, query: dict[str, str], evidence: dict[str, str]) -> float:
        """Get the probability of query given evidence
        
            The (key, value) pairs in query are (r.v. name, value for r.v.),
            and similarly for evidence.
        """
        # use get_marginal_prob and the definition of conditional probability to complete this function
        # Do not modify the parameters' values (i.e., do not add or remove from the dictionaries)
        # the definition of conditional probability is:
        # P(A | B) = P(A, B) / P(B)

        # merge query and evidence into one dictionary
        joint_assignment = query.copy() | evidence.copy()

        # get the marginal probability of the query and evidence
        numerator = self.get_marginal_prob(joint_assignment)
        # get the marginal probability of the evidence by itself
        denominator = self.get_marginal_prob(evidence)

        # if the denominator is 0, return 0.0
        if denominator == 0:
            return 0.0

        return numerator / denominator
    

    def get_conditional_prob_distribution(self, query_vars: list[str], evidence: dict[str, str])\
          -> dict[tuple[str, ...], float]:
        """Get the probability distribution of query variables given evidence
        
            query is a list of variable names

            The (key, value) pairs in evidence are (r.v. name, value for r.v.)

            Returns a distribution, a dictionary mapping tuple of query values to a probability.
            
            Example: query = ['A', 'B'], and let "res" be the returned dictionary.
            Then res[ (t, 7) ] is P(A=t, B=7|evidence)
        """
        # Do NOT use get_marginal_prob to directly compute the probability of evidence
        # (i.e., the denominator in the conditional probability definition formula).
        # Instead, compute the numerator for each part of the distribution
        # (i.e., prob. of query AND evidence for each possible values for query variables),
        # then normalize those numerators so they sum to 1.0.

        # Step 1: Build result so result[query_val_tuple] is the probability
        #   P(query_vars = query_val_tuple, evidence)
        result: dict[tuple[str, ...], float] = {}

        query_vals = self.enumerate_variables_tuples(query_vars)
        for query_val_tuple in query_vals:
            # query_val_tuple is one possible assignment for the query variables
            # The order corresponds to that of query_vars.
            # For example, if query_vars is ['A', 'B', 'C'],
            # then query_val_tuple could be (1, 4, 'shoe'), corresponding to
            # A=1, B=4, C=shoe

            # to set result[query_val_tuple] to be P(query_vars = query_val_tuple, evidence)

            # Create a temporary dictionary 'joint_assignment' by combining 'query_vars' and 'query_val_tuple' with 'evidence'.
            # This dictionary represents one possible joint assignment of values to query variables and evidence variables.
            joint_assignment: dict[str, str] = {}

            for i in range(len(query_vars)):
                #for each variable in query_vars, add the corresponding value from query_val_tuple
                joint_assignment[query_vars[i]] = query_val_tuple[i]

            for var, val in evidence.items():
                #for each variable in evidence, add the corresponding value from evidence
                joint_assignment[var] = val

            # Calculate the joint probability of this assignment using get_marginal_prob
            # P(query_vars = query_val_tuple, evidence)
            result[query_val_tuple] = self.get_marginal_prob(joint_assignment)



        # Now the sum of all the values in result is P(evidence)
        # = sum over query_val_tuple of P(query_vars = query_val_tuple, evidence)

        # Step 2: normalize the values in result (i.e., sum the values,
        #  then divide each value by the sum)
        # Get the sum of all the values in result
        sum_result = sum(result.values())
        for query_val_tuple in result:
            # Divide each value by the sum
            if sum_result != 0:
                result[query_val_tuple] /= sum_result
            else:
                result[query_val_tuple] = 0.0

        # return the final answer
        return result


    def __cond_prob_topo_order(self, query: dict[str, str], evidence: dict[str, str], next_var_index: int) -> float:
        """ Compute a conditional probability (satisfying a restricted form) using the chain rule
         
            Let v be the variable topo_order[next_var_index].
            Every variable in the evidence must come before v in topo_order.
        
            This function returns P(q' | e), where
              q' is the subset of query for variables at or after v in topo_order,
              and e is the evidence.

            Two cases:
            1. v is in query with value x
               Then P(q' | e) = P(v=x, q'' | e) = P(v=x | e) P(q''| e, v=x),
               where q'' is q' without v.
            2. v is not in query
               Then P(q' | e) = {sum over all x in v's domain of P(v=x, q'| e)}
               P(v=x, q'|e) is equal to P(v=x|e) * P(q'|e, v=x)
        """

        if next_var_index >= len(self.topo_order):
            return 1.0

        next_var = self.topo_order[next_var_index]
        node = self.nodes[next_var]

        if next_var in query:
            # Don't branch on possible values for next_var; just use the one from the query
            evidence[next_var] = query[next_var]
            prob = (node.get_probability(query[next_var], self.get_parent_values(node.name, evidence)) *
                    self.__cond_prob_topo_order(query, evidence, next_var_index + 1))
            del evidence[next_var]
        else:
            # Condition on possible values for next_var
            # At the end of this else block, prob should be the SUM over all x in next_var's domain of
            #    P(next_var=x | parents' values) * P(later vars in topo order| next_var=x, earlier vars in topo order)
            # Note that the "earlier vars in topo order" are those in the evidence variable right now
            prob = 0.0

            for x in node.values:

                # P(next_var=x | parents' values) * P(later vars in topo order| next_var=x, earlier vars in topo order)
                evidence[next_var] = x
                prob = prob + (node.get_probability(x, self.get_parent_values(node.name, evidence)) *
                    self.__cond_prob_topo_order(query, evidence, next_var_index + 1))
                                # Hint: Set the appropriate evidence, then use recursion to compute the second term in the product
                pass
            
            del evidence[next_var]

        return prob
    

#####################   Exact inference is above

#####################   Approximate inference is below

    def get_markov_blanket(self, var_name: str) -> list[str]:
        """Returns a list of the variable names in the Markov blanket of var_name

            Should always return the variables in the same order
        """

        # Build markov_blanket_vars to be the names of the variables in
        # the Markov blanket of var_name: its parents, children, and childrens' parents.
        # Note that a node could be a parent of both var_name and one of var_name's children,
        # but it should only be included once.
        markov_blanket_vars = []

        # Add parents
        markov_blanket_vars.extend(self.nodes[var_name].parents)

        # Find children
        children = self.get_children(var_name)
        markov_blanket_vars.extend(children)

        # Add other parents of children (spouses), avoiding duplicates
        for child in children:
            for parent in self.nodes[child].parents:
                if parent != var_name and parent not in markov_blanket_vars:
                    markov_blanket_vars.append(parent)


        # remove duplicates while preserving order (may not be necessary)
        # markov_blanket_vars = self.remove_duplicates_preserve_order(markov_blanket_vars)
    
        # DONE: complete the function to compute and return the markov blanket variables' names
        return markov_blanket_vars

    def get_children(self, var_name: str) -> list[str]:
        """Returns a list of the variable names that are children of var_name"""
        children = list()
        for node_name, node in self.nodes.items():
            if var_name in node.parents:
                children.append(node_name)

        return children

    def approx_conditional_prob_distribution(self, method: str, num_samples: int,
                                             query_vars: list[str], evidence: dict[str, str],
                                             gibbs: "Optional[GibbsSampler]" = None, rand_seed: Optional[int] = None) -> dict[tuple, float]:
        """Approximate the probability distribution of query variables given evidence, using sampling

            method should be one of "rejection", "likelihood_weighting", or "gibbs"
        
            query is a list of variable names

            The (key, value) pairs in evidence are (r.v. name, value for r.v.)

            Returns a distribution, a dictionary mapping each possible tuple of query values to a probability.
            The sum of the values will be 1.0.
            
            Example: query_vars = ['A', 'B'], and let "res" be the returned dictionary.
            Then res[ (t, 7) ] is an estimate for P(A=t, B=7|evidence)

            gibbs is ignored for method != "gibbs"; otherwise, it should be a GibbsSampler built for this network
        """

        # NOTE that this method is not well-structured in terms of software engineering,
        # but it is organized this way to section off parts of code for you to complete.

        # We generate GIBBS_BURNIN samples for a Gibbs sampler before we start counting.
        # That burnin gives the sampler time to "move" toward values more consistent with the evidence.
        GIBBS_BURNIN = 100

        random.seed(rand_seed)
        
        # To start, each possible query_val_tuple has weight 0.0.
        # As we generate samples that match, we add to that weight.
        sample_weight = {}
        query_vals = self.enumerate_variables_tuples(query_vars)
        for query_val_tuple in query_vals:
            sample_weight[query_val_tuple] = 0.0

        if method == 'rejection':
            # At the end of this block, sample_weight[v] will be
            # the number of samples where the query variables had values v

            for i in range(num_samples):
                sample, _ = self.__gen_sample(evidence)
                query_of_sample = tuple([sample[rv] for rv in query_vars])
                sample_weight[query_of_sample] += 1.0

        elif method == 'likelihood_weighting':
            # implement likelihood weighting, using num_samples samples.
            #
            # At the end of this block, sample_weight[v] should be
            # the total weight of the samples where the query variables had values
            for _ in range(num_samples):
                sample, weight = self.__gen_sample(evidence, True)  # Get both sample and weight
                query_of_sample = tuple([sample[rv] for rv in query_vars])
                sample_weight[query_of_sample] += weight  # Add the weight to the accumulator

        elif method == 'gibbs':
            assert(gibbs is not None)

            # The _ is because we don't need the weight returned by __gen_sample
            sample, _ = self.__gen_sample(evidence, True)
            
            # Gibbs sampling needs to "burn in" by generating a few samples before we start counting
            non_evidence_vars = [x for x in self.topo_order if x not in evidence]
            for _ in range(GIBBS_BURNIN):
                gibbs.update_sample(sample, non_evidence_vars)

            # Now we can start counting

            for i in range(num_samples):
                # Update the sample using the Gibbs sampler
                gibbs.update_sample(sample, non_evidence_vars)

                # Get the values of the query variables in the sample
                query_of_sample = tuple([sample[rv] for rv in query_vars])

                # Increment the weight for that query value
                sample_weight[query_of_sample] += 1.0

            pass

        else:
            raise ValueError(f"Invalid sampling method: {method}")

        
        # normalize the sample_weight dictionary,
        # dividing each value by the sum of the original values,
        # so the new sample_weight values sum to 1.0

        # Get the sum of all the values in sample_weight
        sum_sample_weight = sum(sample_weight.values())
        for query_val_tuple in sample_weight:
            # Divide each value by the sum
            if sum_sample_weight != 0:
                sample_weight[query_val_tuple] /= sum_sample_weight
            else:
                sample_weight[query_val_tuple] = 0.0

        # Return the final result
        return sample_weight
    

    def __gen_sample(self, evidence: dict[str,str], use_likelihood_weighting=False) -> tuple[dict[str,str], float]:
        """Returns one sample (map from variable name to value) of all network variables and a weight for the sample.

            If use_likelihood_weighting is false, use rejection sampling until the
            resulting sample is consistent with the evidence. The weight is always 1.0.

            If use_likelihood_weighting is true, use likelihood weighting to avoid
            sampling the evidence variables. The weight varies by sample according to the likelihood weighting algorithm.
        """

        if not use_likelihood_weighting:
            # Rejection sampling
            weight = 1.0

            # Sample until we get a sample consistent with the evidence
            while True:

                # generate a sample
                sample = {}
                for var_name in self.topo_order:
                    sample[var_name] = self.nodes[var_name].sample_value(
                        self.get_parent_values(var_name, sample))

                # Compute the weight of the sample

                #CHECK SAMPLE
                sample_is_consistent = True
                # for each evidence variable
                for ev_name in evidence:
                    #if the sample disagrees with the evidence
                    if sample.get(ev_name) != evidence[ev_name]:
                        sample_is_consistent = False
                        break

                if sample_is_consistent:
                    # If the sample is consistent with the evidence, return it
                    return sample, weight

                #otherwise continue sampling

        else:

            # Likelihood weighted sampling
            weight = 1.0

            # generate a sample
            sample = {}
            for var_name in self.topo_order:
                if var_name in evidence:
                    # If the variable is evidence, set its value
                    sample[var_name] = evidence[var_name]
                    # and multiply the weight by the probability of that value occuring given it's parents.
                    weight *= self.nodes[var_name].get_probability(
                        sample[var_name], self.get_parent_values(var_name, sample))
                else:
                    sample[var_name] = self.nodes[var_name].sample_value(
                        self.get_parent_values(var_name, sample))

            return sample, weight

       #if we get here, something went horribly wrong.
        #return {}, 1.0


class GibbsSampler:
    """ Facilitates Gibbs sampling of a network, precomputing nodes' probabilities
        given their Markov blankets
    """

    def __init__(self, network: BayesianNetwork):
        # gibbs_tables[v][blanket_vals] is the distribution of the variable with name v,
        # given the blanket_vals values for the markov blanket

        self.network = network

        self.gibbs_tables: dict[str, dict[tuple, dict[tuple, float]]] = {}

        for v in network.topo_order:
            self.gibbs_tables[v] = {}
            markov_blanket_vars = network.get_markov_blanket(v)

            markov_blanket_val_tuples = network.enumerate_variables_tuples(markov_blanket_vars)

            for blanket_vals in markov_blanket_val_tuples:
                # Explicitly construct the evidence dictionary to ensure correct order
                evidence = {}
                for i, var_name in enumerate(markov_blanket_vars):
                    evidence[var_name] = blanket_vals[i]

                self.gibbs_tables[v][blanket_vals] = network.get_conditional_prob_distribution([v], evidence)

    def update_sample(self, sample: dict[str, str], non_evidence_vars: list[str]):
        """ Randomly generate the next Gibbs sample, altering one variable in the given sample

            non_evidence_vars are all the variables in the network except the evidence variables
            (i.e., the variables that we sample, instead of clamping to certain values)
        """

        # select a random non evidence variable
        variable_to_update = random.choice(non_evidence_vars)

        # get the names of the markov blanket variables
        markov_blanket_vars = self.network.get_markov_blanket(variable_to_update)
        markov_blanket_values = {}

        # for each blanket variable, put it in the dictionary as a key to it's current value in the sample
        for blanket_var in markov_blanket_vars:
            markov_blanket_values[blanket_var] = sample[blanket_var]

        # store the values in a tuple to use as a key in the gibbs_tables to get the precomputed distribution
        # Explicitly construct the tuple to ensure correct order
        markov_blanket_values_tuple = tuple(markov_blanket_values[var] for var in markov_blanket_vars)

        # get the distribution for the variable to update, given the markov blanket values
        distribution = self.gibbs_tables[variable_to_update][markov_blanket_values_tuple]
        # select a new value for the variable to update, using the distribution
        new_value = random.choices(list(distribution.keys()), list(distribution.values()))[0]

        # finally, update the value in our sample to the random value we just selected
        sample[variable_to_update] = new_value[0]


