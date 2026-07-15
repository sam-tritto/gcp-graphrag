import html
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def build_bayesian_model(blueprint_records, user_stats):
    """
    Dynamically constructs a Bayesian Belief Network from active exam blueprint.
    
    Structure: Domain_Domain -> Service_Service -> SubConcept_SubConcept -> SubConcept_Question
    
    Args:
        blueprint_records: list of dicts with keys:
            "domain": domain name
            "service": service name (optional, new format)
            "subconcepts": list of sub-concept names (optional, new format)
            "services": list of service names (optional, fallback format)
        user_stats: dict with keys "domains", "services", and "subconcepts", holding alpha/beta values
        
    Returns:
        model: initialized and validated DiscreteBayesianNetwork instance
        latent_nodes: list of domain, service, and subconcept node names
        candidate_questions: list of question node names
    """
    edges = []
    domain_nodes = set()
    service_nodes = set()
    subconcept_nodes = set()
    question_nodes = set()
    
    # Track domain-service-subconcept mapping to construct edge list
    for record in blueprint_records:
        d_name = record["domain"]
        d_node = f"{d_name}_Domain"
        domain_nodes.add(d_node)
        
        s_name = record.get("service")
        if not s_name:
            # Fallback if old format is passed in tests
            for s_old in record.get("services", []):
                s_node = f"{s_old}_Service"
                sub_node = f"{s_old}_SubConcept"
                q_node = f"{s_old}_Question"
                service_nodes.add(s_node)
                subconcept_nodes.add(sub_node)
                question_nodes.add(q_node)
                edges.append((d_node, s_node))
                edges.append((s_node, sub_node))
                edges.append((sub_node, q_node))
            continue
            
        s_node = f"{s_name}_Service"
        service_nodes.add(s_node)
        edges.append((d_node, s_node))
        
        for sub_name in record.get("subconcepts", []):
            if not sub_name:
                continue
            sub_node = f"{sub_name}_SubConcept"
            q_node = f"{sub_name}_Question"
            subconcept_nodes.add(sub_node)
            question_nodes.add(q_node)
            
            # Service -> SubConcept
            edges.append((s_node, sub_node))
            # SubConcept -> Question
            edges.append((sub_node, q_node))
            
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
            cols = 2**k
            row_fail = []
            row_pass = []
            for c in range(cols):
                set_bits = 0
                for bit_idx in range(k):
                    if (c >> bit_idx) & 1:
                        set_bits += 1
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

    # 3. Define SubConcept CPDs conditional on their parent services
    for sub_node in subconcept_nodes:
        parents = sorted(list(model.get_parents(sub_node)))
        k = len(parents)
        
        if k > 0:
            cols = 2**k
            row_fail = []
            row_pass = []
            for c in range(cols):
                set_bits = 0
                for bit_idx in range(k):
                    if (c >> bit_idx) & 1:
                        set_bits += 1
                p_mastery = 0.1 + 0.7 * (set_bits / k)
                row_pass.append(p_mastery)
                row_fail.append(1.0 - p_mastery)
                
            cpd = TabularCPD(
                variable=sub_node,
                variable_card=2,
                values=[row_fail, row_pass],
                evidence=parents,
                evidence_card=[2] * k
            )
        else:
            sub_name = sub_node.replace("_SubConcept", "")
            stats = user_stats.get("subconcepts", {}).get(sub_name, {"alpha": 1, "beta": 1})
            alpha = stats.get("alpha", 1)
            beta = stats.get("beta", 1)
            p_mastery = beta / (alpha + beta)
            cpd = TabularCPD(
                variable=sub_node,
                variable_card=2,
                values=[[1.0 - p_mastery], [p_mastery]]
            )
            
        cpds.append(cpd)
        
    # 4. Define Question CPDs conditional on SubConcept mastery
    for q_node in question_nodes:
        sub_node = q_node.replace("_Question", "_SubConcept")
        # Fail/Pass conditional on SubConcept: P(Q=1 | SC=0) = 0.05, P(Q=1 | SC=1) = 0.90
        cpd = TabularCPD(
            variable=q_node,
            variable_card=2,
            values=[
                [0.95, 0.10],  # Q=0 (Fail) given SC=0, SC=1
                [0.05, 0.90]   # Q=1 (Pass) given SC=0, SC=1
            ],
            evidence=[sub_node],
            evidence_card=[2]
        )
        cpds.append(cpd)
        
    # Bind CPDs to the network and validate DAG constraints
    model.add_cpds(*cpds)
    model.check_model()
    
    latent_nodes = sorted(list(domain_nodes) + list(service_nodes) + list(subconcept_nodes))
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
    Selects the next question (subconcept_name) that maximizes information gain
    (reduces expected Shannon entropy of all latent nodes the most).
    """
    best_subconcept = None
    min_expected_entropy = float("inf")
    
    # 1. Prune Candidate Space: Only evaluate candidate questions for subconcepts
    # belonging to services where the user's current mastery is highly uncertain (entropy > 0.5).
    service_nodes = [node for node in latent_nodes if node.endswith("_Service")]
    service_entropies = {}
    uncertain_services = set()
    
    for s_node in service_nodes:
        query_res = inference.query(variables=[s_node], show_progress=False)
        ent = calculate_entropy(query_res.values)
        service_entropies[s_node] = ent
        if ent > 0.5:
            uncertain_services.add(s_node)
            
    # Fallback to service(s) with the highest entropy if none exceed 0.5
    if not uncertain_services and service_entropies:
        max_ent = max(service_entropies.values())
        uncertain_services = {s for s, ent in service_entropies.items() if ent >= max_ent - 1e-9}
        
    # Filter candidate questions to subconcepts belonging to uncertain services
    pruned_candidates = []
    for q_node in candidate_questions:
        sub_node = q_node.replace("_Question", "_SubConcept")
        if sub_node in model.nodes():
            parents = model.get_parents(sub_node)
            if any(parent in uncertain_services for parent in parents):
                pruned_candidates.append(q_node)
        else:
            pruned_candidates.append(q_node)
            
    if not pruned_candidates:
        pruned_candidates = candidate_questions
        
    # 2. Evaluate candidate questions with approximate entropy calculation
    for q_node in pruned_candidates:
        subconcept_name = q_node.replace("_Question", "")
        
        # Query BBN for probability of passing this question
        query_res = inference.query(variables=[q_node], show_progress=False)
        p_pass = query_res.values[1]
        p_fail = 1.0 - p_pass
        
        # Approximate Entropy: Only query a subset of highly connected latent nodes
        # (e.g. the subconcept node itself and its parent services/domains)
        sub_node = q_node.replace("_Question", "_SubConcept")
        local_latent_nodes = [sub_node]
        if sub_node in model.nodes():
            parents = model.get_parents(sub_node)
            local_latent_nodes.extend(list(parents))
            for p in parents:
                if p in model.nodes():
                    local_latent_nodes.extend(list(model.get_parents(p)))
        local_latent_nodes = [n for n in local_latent_nodes if n in latent_nodes]
        
        if not local_latent_nodes:
            local_latent_nodes = latent_nodes
            
        # Simulate outcomes and calculate updated local network entropy
        entropy_if_passed = calculate_entropy(simulate_evidence(model, inference, local_latent_nodes, q_node, pass_state=1))
        entropy_if_failed = calculate_entropy(simulate_evidence(model, inference, local_latent_nodes, q_node, pass_state=0))
        
        # Calculate Expected Entropy
        expected_entropy = (p_pass * entropy_if_passed) + (p_fail * entropy_if_failed)
        
        if expected_entropy < min_expected_entropy:
            min_expected_entropy = expected_entropy
            best_subconcept = subconcept_name
            
    return best_subconcept

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

def generate_bbn_dot_graph(blueprint_records, user_stats, inference=None, bbn_model=None, show_subconcepts=True, domain_filter=None, rankdir="TB"):
    """
    Constructs a DOT language string representing the Domain-Service-SubConcept hierarchy.
    Colors nodes dynamically based on current mastery (posterior or prior).
    """
    dot_lines = [
        "digraph G {",
        f'    graph [bgcolor="transparent", rankdir="{rankdir}", splines="true", nodesep="0.5", ranksep="0.6", margin="0"];',
        '    node [fontname="sans-serif", fontsize="10", style="filled,rounded", penwidth="1.5", margin="0.15,0.1"];',
        '    edge [color="#9aa0a6", arrowsize="0.8", penwidth="1.2"];',
        ""
    ]
    
    rendered_nodes = set()
    edges = []
    
    for record in blueprint_records:
        d_name = record["domain"]
        if domain_filter and domain_filter != "All Domains" and d_name != domain_filter:
            continue
            
        d_node = f"{d_name}_Domain"
        
        # Calculate Domain mastery
        p_domain = 0.5
        if inference and bbn_model and d_node in bbn_model.nodes():
            try:
                p_domain = float(inference.query(variables=[d_node], show_progress=False).values[1])
            except Exception:
                d_stats = user_stats.get("domains", {}).get(d_name, {"alpha": 1, "beta": 1})
                p_domain = d_stats["beta"] / (d_stats["alpha"] + d_stats["beta"])
        else:
            d_stats = user_stats.get("domains", {}).get(d_name, {"alpha": 1, "beta": 1})
            p_domain = d_stats["beta"] / (d_stats["alpha"] + d_stats["beta"])
            
        # Style based on mastery
        if p_domain >= 0.7:
            d_fill = "#81c995"  # Soft Green
            d_border = "#137333"
        elif p_domain >= 0.3:
            d_fill = "#fdd663"  # Soft Yellow
            d_border = "#b06000"
        else:
            d_fill = "#f28b82"  # Soft Red
            d_border = "#c5221f"
            
        d_label_escaped = html.escape(d_name.upper())
        d_label = f"<<b>📚 {d_label_escaped}</b><br/><font point-size='9'>Mastery: {p_domain:.1%}</font>>"
        
        if d_node not in rendered_nodes:
            dot_lines.append(f'    "{d_node}" [label={d_label}, fillcolor="{d_fill}", color="{d_border}", fontcolor="#202124", shape="box", penwidth="2.5"];')
            rendered_nodes.add(d_node)
            
        # Check new vs fallback record formats
        services_map = {}
        if "service" in record:
            services_map = {record["service"]: record.get("subconcepts", [])}
        elif "services" in record:
            services_map = {s: [s] for s in record["services"]}
            
        for s_name, subconcepts in services_map.items():
            if not s_name:
                continue
            s_node = f"{s_name}_Service"
            
            # Calculate Service mastery
            p_service = 0.5
            if inference and bbn_model and s_node in bbn_model.nodes():
                try:
                    p_service = float(inference.query(variables=[s_node], show_progress=False).values[1])
                except Exception:
                    s_stats = user_stats.get("services", {}).get(s_name, {"alpha": 1, "beta": 1})
                    p_service = s_stats["beta"] / (s_stats["alpha"] + s_stats["beta"])
            else:
                s_stats = user_stats.get("services", {}).get(s_name, {"alpha": 1, "beta": 1})
                p_service = s_stats["beta"] / (s_stats["alpha"] + s_stats["beta"])
                
            # Style based on mastery
            if p_service >= 0.7:
                s_fill = "#81c995"
                s_border = "#137333"
            elif p_service >= 0.3:
                s_fill = "#fdd663"
                s_border = "#b06000"
            else:
                s_fill = "#f28b82"
                s_border = "#c5221f"
                
            s_label_escaped = html.escape(s_name)
            s_label = f"<☁️ {s_label_escaped}<br/><font point-size='9'>Mastery: {p_service:.1%}</font>>"
            
            if s_node not in rendered_nodes:
                dot_lines.append(f'    "{s_node}" [label={s_label}, fillcolor="{s_fill}", color="{s_border}", fontcolor="#202124", shape="box"];')
                rendered_nodes.add(s_node)
                
            edges.append(f'    "{d_node}" -> "{s_node}";')
            
            # Subconcepts
            if show_subconcepts:
                for sub_name in subconcepts:
                    if not sub_name:
                        continue
                    sub_node = f"{sub_name}_SubConcept"
                    
                    # Calculate SubConcept mastery
                    p_sub = 0.5
                    if inference and bbn_model and sub_node in bbn_model.nodes():
                        try:
                            p_sub = float(inference.query(variables=[sub_node], show_progress=False).values[1])
                        except Exception:
                            sub_stats = user_stats.get("subconcepts", {}).get(sub_name, {"alpha": 1, "beta": 1})
                            p_sub = sub_stats["beta"] / (sub_stats["alpha"] + sub_stats["beta"])
                    else:
                        sub_stats = user_stats.get("subconcepts", {}).get(sub_name, {"alpha": 1, "beta": 1})
                        p_sub = sub_stats["beta"] / (sub_stats["alpha"] + sub_stats["beta"])
                        
                    # Style SubConcept node (Soft light purple/blue for subconcepts)
                    if p_sub >= 0.7:
                        sub_fill = "#d2e3fc"
                        sub_border = "#1a73e8"
                    elif p_sub >= 0.3:
                        sub_fill = "#feefc3"
                        sub_border = "#f9ab00"
                    else:
                        sub_fill = "#fce8e6"
                        sub_border = "#d93025"
                        
                    sub_label_escaped = html.escape(sub_name)
                    sub_label = f"<⚙️ {sub_label_escaped}<br/><font point-size='8'>Mastery: {p_sub:.1%}</font>>"
                    
                    if sub_node not in rendered_nodes:
                        dot_lines.append(f'    "{sub_node}" [label={sub_label}, fillcolor="{sub_fill}", color="{sub_border}", fontcolor="#202124", shape="box"];')
                        rendered_nodes.add(sub_node)
                        
                    edges.append(f'    "{s_node}" -> "{sub_node}";')
                
    # Deduplicate edges
    edges = sorted(list(set(edges)))
    dot_lines.extend(edges)
    dot_lines.append("}")
    return "\n".join(dot_lines)
