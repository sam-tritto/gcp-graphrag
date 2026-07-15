import pytest
import numpy as np
from pgmpy.inference import VariableElimination
from utils.bayesian_engine import (
    build_bayesian_model,
    calculate_entropy,
    select_question_active_learning,
    select_domain_thompson_sampling,
    select_service_thompson_sampling,
    get_overall_entropy,
    generate_bbn_dot_graph
)

def test_build_bayesian_model_success():
    blueprint_records = [
        {"domain": "Compute", "services": ["GCE", "GKE"]},
        {"domain": "Storage", "services": ["GCS"]}
    ]
    user_stats = {
        "domains": {
            "Compute": {"alpha": 1, "beta": 1},
            "Storage": {"alpha": 2, "beta": 1}
        },
        "services": {
            "GCE": {"alpha": 1, "beta": 1},
            "GKE": {"alpha": 1, "beta": 2},
            "GCS": {"alpha": 1, "beta": 1}
        }
    }
    
    model, latent_nodes, candidate_questions = build_bayesian_model(blueprint_records, user_stats)
    
    # Assert nodes are created as expected
    assert "Compute_Domain" in latent_nodes
    assert "Storage_Domain" in latent_nodes
    assert "GCE_Service" in latent_nodes
    assert "GKE_Service" in latent_nodes
    assert "GCS_Service" in latent_nodes
    
    assert "GCE_Question" in candidate_questions
    assert "GKE_Question" in candidate_questions
    assert "GCS_Question" in candidate_questions
    
    # Check that it compiles and validates against pgmpy constraints
    assert model.check_model() is True

def test_build_bayesian_model_empty():
    model, latent_nodes, candidate_questions = build_bayesian_model([], {})
    assert len(latent_nodes) == 0
    assert len(candidate_questions) == 0
    assert len(model.nodes()) == 0

def test_calculate_entropy():
    # Maximum uncertainty (equal split)
    max_ent = calculate_entropy([0.5, 0.5])
    assert abs(max_ent - 1.0) < 1e-9
    
    # Perfect certainty (100% mastery)
    zero_ent = calculate_entropy([1.0, 0.0])
    assert abs(zero_ent - 0.0) < 1e-9
    
    # Semi-certainty
    some_ent = calculate_entropy([0.9, 0.1])
    assert 0.0 < some_ent < 1.0

def test_select_domain_thompson_sampling():
    domain_stats = {
        "DomainA": {"alpha": 1, "beta": 1},
        "DomainB": {"alpha": 2, "beta": 5}
    }
    next_domain, scores = select_domain_thompson_sampling(domain_stats)
    assert next_domain in domain_stats
    assert len(scores) == 2
    assert "DomainA" in scores
    assert "DomainB" in scores

def test_select_service_thompson_sampling():
    service_stats = {
        "ServiceA": {"alpha": 2, "beta": 2},
        "ServiceB": {"alpha": 1, "beta": 5}
    }
    next_service = select_service_thompson_sampling(service_stats, ["ServiceA", "ServiceB"])
    assert next_service in ["ServiceA", "ServiceB"]

def test_active_learning_and_entropy_calculations():
    blueprint_records = [
        {"domain": "Compute", "services": ["GCE"]}
    ]
    user_stats = {
        "domains": {
            "Compute": {"alpha": 1, "beta": 1}
        },
        "services": {
            "GCE": {"alpha": 1, "beta": 1}
        }
    }
    
    model, latent_nodes, candidate_questions = build_bayesian_model(blueprint_records, user_stats)
    inference = VariableElimination(model)
    
    # 1. Total Entropy
    overall_ent = get_overall_entropy(model, inference, latent_nodes)
    assert overall_ent > 0.0
    
    # 2. Active learning selection should select the only available service
    best_service = select_question_active_learning(model, inference, latent_nodes, candidate_questions)
    assert best_service == "GCE"

def test_generate_bbn_dot_graph():
    blueprint_records = [
        {"domain": "Compute", "services": ["GCE"]}
    ]
    user_stats = {
        "domains": {
            "Compute": {"alpha": 1, "beta": 1}
        },
        "services": {
            "GCE": {"alpha": 1, "beta": 1}
        }
    }
    model, latent_nodes, candidate_questions = build_bayesian_model(blueprint_records, user_stats)
    inference = VariableElimination(model)
    
    dot_str = generate_bbn_dot_graph(blueprint_records, user_stats, inference, model)
    assert "digraph G {" in dot_str
    assert "Compute_Domain" in dot_str
    assert "GCE_Service" in dot_str
    assert "Mastery:" in dot_str
