import pytest
from sync_pipeline import GCP_EXAMS
from utils.bayesian_engine import build_bayesian_model

def test_all_certification_bbn_models():
    """Verify that every certification config compiles successfully into a BBN model."""
    for exam_id, config in GCP_EXAMS.items():
        blueprint_records = []
        for domain, services in config["domains"].items():
            blueprint_records.append({
                "domain": domain,
                "services": services
            })
            
        user_stats = {
            "domains": {d: {"alpha": 1, "beta": 1} for d in config["domains"]},
            "services": {s: {"alpha": 1, "beta": 1} for d in config["domains"] for s in config["domains"][d]}
        }
        
        model, latent_nodes, candidate_questions = build_bayesian_model(blueprint_records, user_stats)
        
        # Verify basic model structure
        assert len(latent_nodes) > 0, f"No latent nodes found for {exam_id}"
        assert len(candidate_questions) > 0, f"No candidate questions found for {exam_id}"
        
        # Verify network consistency with pgmpy checks
        assert model.check_model() is True, f"BBN validation failed for certification: {exam_id}"
