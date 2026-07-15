import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def build_bayesian_model(blueprint_records, user_stats):
    """
    Dynamically constructs a Bayesian Belief Network from active exam blueprint.
    
    Structure: Domain_Domain -> Service_Service -> Service_Question
    
    Args:
        blueprint_records: list of dicts with keys "domain" and "services" (list of service names)
        user_stats: dict with keys "domains" and "services", holding alpha/beta values
        
    Returns:
        model: initialized and validated DiscreteBayesianNetwork instance
        latent_nodes: list of domain and service node names
        candidate_questions: list of question node names
    """
    edges = []
    domain_nodes = set()
    service_nodes = set()
    question_nodes = set()
    
    # Track domain-service mapping to construct edge list
    for record in blueprint_records:
        d_name = record["domain"]
        d_node = f"{d_name}_Domain"
        domain_nodes.add(d_node)
        
        for s_name in record["services"]:
            if not s_name:
                continue
            s_node = f"{s_name}_Service"
            q_node = f"{s_name}_Question"
            service_nodes.add(s_node)
            question_nodes.add(q_node)
            
            # Domain -> Service
            edges.append((d_node, s_node))
            # Service -> Question
            edges.append((s_node, q_node))
            
    # Remove duplicate edges
    edges = list(set(edges))
    
    if not edges:
        # Fallback empty model helper
        model = DiscreteBayesianNetwork()
        return model, [], []
        
    model = DiscreteBayesianNetwork(edges)
    
    cpds = []
    
    # 1. Define Domain prior CPDs (root nodes)
    for d_node in domain_nodes:
        d_name = d_node.replace("_Domain", "")
        stats = user_stats.get("domains", {}).get(d_name, {"alpha": 1, "beta": 1})
        alpha = stats.get("alpha", 1)
        beta = stats.get("beta", 1)
        # Prior mastery probability (State 1) from Beta distribution mean
        p_mastery = beta / (alpha + beta)
        
        cpd = TabularCPD(
            variable=d_node,
            variable_card=2,
            values=[[1.0 - p_mastery], [p_mastery]]
        )
        cpds.append(cpd)
        
    # 2. Define Service CPDs conditional on their parent domains
    for s_node in service_nodes:
        parents = sorted(list(model.get_parents(s_node)))
        k = len(parents)
        
        if k > 0:
            # Generate conditional probability table of shape [2, 2^k]
            cols = 2**k
            row_fail = []
            row_pass = []
            for c in range(cols):
                # Count how many parents are in active state 1
                set_bits = 0
                for bit_idx in range(k):
                    if (c >> bit_idx) & 1:
                        set_bits += 1
                
                # Formula: base 10% mastery probability, +70% mastery scaled linearly by parents mastered
                p_mastery = 0.1 + 0.7 * (set_bits / k)
                row_pass.append(p_mastery)
                row_fail.append(1.0 - p_mastery)
                
            cpd = TabularCPD(
                variable=s_node,
                variable_card=2,
                values=[row_fail, row_pass],
                evidence=parents,
                evidence_card=[2] * k
            )
        else:
            # Fallback service prior if it somehow has no parents
            s_name = s_node.replace("_Service", "")
            stats = user_stats.get("services", {}).get(s_name, {"alpha": 1, "beta": 1})
            alpha = stats.get("alpha", 1)
            beta = stats.get("beta", 1)
            p_mastery = beta / (alpha + beta)
            cpd = TabularCPD(
                variable=s_node,
                variable_card=2,
                values=[[1.0 - p_mastery], [p_mastery]]
            )
            
        cpds.append(cpd)
        
    # 3. Define Question CPDs conditional on Service mastery
    for q_node in question_nodes:
        s_node = q_node.replace("_Question", "_Service")
        # Fail/Pass conditional on Service: P(Q=1 | S=0) = 0.05, P(Q=1 | S=1) = 0.90
        # values: row 0 (Fail Q=0), row 1 (Pass Q=1)
        cpd = TabularCPD(
            variable=q_node,
            variable_card=2,
            values=[
                [0.95, 0.10],  # Q=0 (Fail) given S=0, S=1
                [0.05, 0.90]   # Q=1 (Pass) given S=0, S=1
            ],
            evidence=[s_node],
            evidence_card=[2]
        )
        cpds.append(cpd)
        
    # Bind CPDs to the network and validate DAG constraints
    model.add_cpds(*cpds)
    model.check_model()
    
    latent_nodes = sorted(list(domain_nodes) + list(service_nodes))
    candidate_questions = sorted(list(question_nodes))
    
    return model, latent_nodes, candidate_questions

def calculate_entropy(probabilities):
    """Calculates Shannon Entropy over a list of state probabilities."""
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

def simulate_evidence(model, inference, latent_nodes, question_node, pass_state):
    """Simulates the outcome of a question node (0 or 1) and returns marginals of all latent nodes."""
    probs = []
    for node in latent_nodes:
        query_res = inference.query(variables=[node], evidence={question_node: pass_state}, show_progress=False)
        probs.extend(query_res.values) # P(node=0) and P(node=1)
    return probs

def select_question_active_learning(model, inference, latent_nodes, candidate_questions):
    """
    Selects the next question (service_name) that maximizes information gain
    (reduces expected Shannon entropy of all latent nodes the most).
    """
    best_service = None
    min_expected_entropy = float("inf")
    
    for q_node in candidate_questions:
        service_name = q_node.replace("_Question", "")
        
        # 1. Query BBN for probability of passing this question
        query_res = inference.query(variables=[q_node], show_progress=False)
        p_pass = query_res.values[1]
        p_fail = 1.0 - p_pass
        
        # 2. Simulate outcomes and calculate updated network entropy
        entropy_if_passed = calculate_entropy(simulate_evidence(model, inference, latent_nodes, q_node, pass_state=1))
        entropy_if_failed = calculate_entropy(simulate_evidence(model, inference, latent_nodes, q_node, pass_state=0))
        
        # 3. Calculate Expected Entropy
        expected_entropy = (p_pass * entropy_if_passed) + (p_fail * entropy_if_failed)
        
        if expected_entropy < min_expected_entropy:
            min_expected_entropy = expected_entropy
            best_service = service_name
            
    return best_service

def select_domain_thompson_sampling(domain_stats):
    """
    Selects the next study domain using Thompson Sampling.
    
    domain_stats = {
        "Security": {"alpha": 4, "beta": 2},
        ...
    }
    """
    sampled_scores = {}
    for domain_name, stats in domain_stats.items():
        alpha = stats.get("alpha", 1)
        beta = stats.get("beta", 1)
        # Higher failure rate (alpha) shifts the density to sample higher values,
        # indicating high priority for remediation.
        sampled_scores[domain_name] = np.random.beta(alpha, beta)
        
    next_domain = max(sampled_scores, key=sampled_scores.get)
    return next_domain, sampled_scores

def select_service_thompson_sampling(service_stats, domain_services):
    """
    Selects the next service under the selected domain using Thompson Sampling.
    """
    sampled_scores = {}
    for s_name in domain_services:
        stats = service_stats.get(s_name, {"alpha": 1, "beta": 1})
        alpha = stats.get("alpha", 1)
        beta = stats.get("beta", 1)
        sampled_scores[s_name] = np.random.beta(alpha, beta)
        
    next_service = max(sampled_scores, key=sampled_scores.get)
    return next_service

def get_overall_entropy(model, inference, latent_nodes):
    """Calculates the current overall entropy of all latent nodes in the BBN."""
    probs = []
    for node in latent_nodes:
        query_res = inference.query(variables=[node], show_progress=False)
        probs.extend(query_res.values)
    return calculate_entropy(probs)
