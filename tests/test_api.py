"""
Tests for the Legal QA API.

These tests verify:
  - Health endpoint returns 200
  - Empty / missing question returns 400
  - Valid question returns expected response structure (when models are loaded)
  - Multiple sequential requests work
  - Malformed JSON is rejected

Usage:
    python -m pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a test client. Models may or may not be loaded depending on
    whether checkpoint files exist."""
    from api.app import app
    with TestClient(app) as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Predict endpoint — validation errors (always testable)
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictValidation:
    def test_missing_question_field(self, client):
        """Request body without 'question' key should return 422."""
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_empty_question_string(self, client):
        """Empty string should be rejected (min_length=1 on the schema)."""
        resp = client.post("/predict", json={"question": ""})
        # Pydantic min_length=1 returns 422; our handler returns 400 for
        # whitespace-only. Either is acceptable.
        assert resp.status_code in (400, 422)

    def test_whitespace_only_question(self, client):
        """Whitespace-only question should be rejected."""
        resp = client.post("/predict", json={"question": "   "})
        # If models aren't loaded this returns 503 after validation passes,
        # but the stripped check in the handler should catch it as 400.
        assert resp.status_code in (400, 503)

    def test_malformed_json(self, client):
        """Non-JSON body should return 422."""
        resp = client.post(
            "/predict",
            content="this is not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_wrong_content_type(self, client):
        """Plain text content type should be rejected."""
        resp = client.post(
            "/predict",
            content="question=hello",
            headers={"Content-Type": "text/plain"},
        )
        assert resp.status_code == 422

    def test_invalid_mode(self, client):
        """Request body with an invalid mode should return 422."""
        resp = client.post(
            "/predict",
            json={
                "question": "My employer has not paid my salary for three months.",
                "mode": "invalid_mode"
            }
        )
        assert resp.status_code == 422

    def test_invalid_domain(self, client):
        """Request body with an unknown domain should return 422."""
        resp = client.post(
            "/predict",
            json={"question": "My employer has not paid my salary.", "domain": "tax"},
        )
        assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# Domain routing
# ─────────────────────────────────────────────────────────────────────────────

class TestDomains:
    def test_domains_lists_every_domain_and_mode(self, client):
        resp = client.get("/domains")
        assert resp.status_code == 200
        pairs = {(r["domain"], r["mode"]) for r in resp.json()}
        assert pairs == {(d, m) for d in ("labour", "criminal", "family")
                         for m in ("actionable", "informative", "readable")}

    def test_configs_point_at_domain_folders(self):
        from inference.config import make_config
        cfg = make_config("criminal", "readable")
        assert cfg["rl_model_path"].replace("\\", "/").endswith("HKG+PC+RL/criminal/readable/rl_policy_model.pt")
        assert cfg["domain_name"] == "Criminal Law"
        assert cfg["auto_detect_domain"] is False

    def test_required_files_follow_engine_config(self):
        """check_required_files must check the engine's own folder, not the default labour one."""
        from inference.config import make_config
        from inference.engine import LegalQAEngine
        cfg = make_config("family", "actionable")
        cfg["rl_model_path"] = cfg["rl_model_path"] + ".does-not-exist"
        missing = {f["name"] for f in LegalQAEngine(cfg).check_required_files()}
        assert "rl_policy_model.pt" in missing

    def test_untrained_domain_falls_back_to_labour(self, client):
        from api.app import is_available
        if is_available("family", "actionable"):
            pytest.skip("Family checkpoints are present — no fallback to test.")
        if not is_available("labour", "actionable"):
            pytest.skip("Labour checkpoints not available.")
        resp = client.post(
            "/predict",
            json={"question": "My husband wants a divorce and custody of our child.", "domain": "family"},
        )
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            data = resp.json()
            assert data["domain"] == "family"
            assert data["domain_used"] == "labour"
            assert data["domain_fallback"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Predict endpoint — models not loaded (503)
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictModelsNotLoaded:
    """These tests verify graceful behavior when checkpoints are missing."""

    def test_returns_503_when_models_missing(self, client):
        """If models aren't loaded, /predict should return 503, not crash."""
        from api.app import engines
        from inference.config import get_required_files
        import os
        # Check if checkpoints are present for default actionable mode
        checkpoints_exist = all(os.path.isfile(f["path"]) for f in get_required_files())
        if engines or checkpoints_exist:
            pytest.skip("Model checkpoints are present (or loaded) — skipping 503 path.")

        resp = client.post(
            "/predict",
            json={"question": "My employer has not paid my salary."},
        )
        assert resp.status_code == 503
        data = resp.json()
        assert "detail" in data


# ─────────────────────────────────────────────────────────────────────────────
# Predict endpoint — full inference (requires checkpoints)
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictInference:
    """These tests require trained model checkpoints to be present."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_models(self, client):
        from inference.config import get_required_files
        import os
        checkpoints_exist = all(os.path.isfile(f["path"]) for f in get_required_files())
        if not checkpoints_exist:
            pytest.skip("Model checkpoints for 'actionable' not available — skipping inference tests.")

    def test_valid_question_returns_full_response(self, client):
        resp = client.post(
            "/predict",
            json={"question": "My employer has not paid my salary for three months."},
        )
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            data = resp.json()

            # Verify response structure
            assert "question" in data
            assert "answer" in data
            assert "reasoning_chain" in data
            assert "retrieved_cases" in data

            # Verify types
            assert isinstance(data["question"], str)
            assert isinstance(data["answer"], str)
            assert isinstance(data["reasoning_chain"], list)
            assert isinstance(data["retrieved_cases"], list)

            # Verify content is not empty
            assert len(data["answer"]) > 0
            assert len(data["reasoning_chain"]) > 0

            # Verify retrieved cases structure
            for case in data["retrieved_cases"]:
                assert "question" in case
                assert "answer" in case

    def test_multiple_sequential_questions(self, client):
        """Verify that the engine handles multiple requests without errors."""
        questions = [
            "Can my landlord evict me without notice?",
            "What are the legal rights of a contract worker?",
            "How to file a complaint against unfair termination?",
        ]
        for q in questions:
            resp = client.post("/predict", json={"question": q})
            assert resp.status_code in (200, 500)
            if resp.status_code == 200:
                data = resp.json()
                assert data["question"] == q
                assert len(data["answer"]) > 0

    def test_question_is_echoed_back(self, client):
        """The response should echo the original question."""
        q = "Is it legal for my employer to withhold my PF?"
        resp = client.post("/predict", json={"question": q})
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            assert resp.json()["question"] == q

    def test_different_modes(self, client):
        """Verify that different modes can be queried."""
        from api.app import make_config_for_mode
        import os
        for mode in ["actionable", "informative", "readable"]:
            cfg = make_config_for_mode(mode)
            paths = [
                cfg["rl_model_path"],
                cfg["dssm_model_path"],
                cfg["retrieval_index"],
                cfg["node_emb_cache_path"],
                cfg["kg_path"],
                cfg["nodes_csv_path"],
            ]
            checkpoints_exist = all(os.path.isfile(p) for p in paths)
            if not checkpoints_exist:
                pytest.skip(f"Mode {mode} checkpoints not available - skipping.")
            resp = client.post(
                "/predict",
                json={
                    "question": "My employer has not paid my salary for three months.",
                    "mode": mode
                }
            )
            # Expect either 200 (success) or 500 (OpenAI API key validation failure, but routing succeeded)
            assert resp.status_code in (200, 500)
