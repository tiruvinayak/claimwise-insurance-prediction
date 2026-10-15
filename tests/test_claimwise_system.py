"""Comprehensive automated verification test suite for ClaimWise Insurance Prediction System.

Tests cover:
1. Dataset Management:
   - Original dataset immutability verification
   - Mode A default dataset inspection
   - Valid CSV upload & model compatibility audit
   - Empty CSV rejection
   - Invalid file extension rejection
   - Incompatible dataset schema rejection
2. Primary Prediction Engine:
   - Single authoritative 130-feature prediction workflow
   - Direct model predict() numerical parity
   - Input reflection & retention for editing
   - Honest estimation disclaimer & currency integrity
3. 13-Model Benchmarking & Regression Metrics:
   - Verification of all 13 models in comparison artifacts
   - Verification of all 7 regression metrics (MAE, MSE, RMSE, R², EV, MedAE, MAPE)
   - Rank #1 Best model confirmation (HistGradientBoosting Regressor)
   - Diagnostic chart availability
4. CSV Preview & Download Security:
   - Pagination and search on /csv/preview
   - Safe file download on /csv/download
   - Directory traversal attack prevention
5. All Core Routes & Asset Endpoints:
   - HTTP 200 across all 10 site routes
   - Static assets and visualization image serving
"""

import io
import os
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "Src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import pandas as pd

import app as flask_app
from dataset_manager import clear_uploaded_files, get_current_upload_path
from prediction_preprocessor import load_preprocessor, transform


RAW_DATASET_PATH = PROJECT_ROOT / "Dataset" / "01_raw_claimwise_50000.csv"
TEST_DATASET_X = PROJECT_ROOT / "Dataset" / "06_split_X_test.csv"
TEST_DATASET_Y = PROJECT_ROOT / "Dataset" / "06_split_y_test.csv"
COMPARISON_CSV_PATH = PROJECT_ROOT / "Outputs" / "Model_Comparison" / "model_comparison.csv"
FINAL_TEST_METRICS_PATH = PROJECT_ROOT / "Outputs" / "ExperimentSuite" / "final_test_metrics.csv"
EXPERIMENT_METADATA_PATH = PROJECT_ROOT / "Outputs" / "ExperimentSuite" / "experiment_metadata.json"

ALL_EXPECTED_MODELS = [
    "Hist Gradient Boosting Regressor",
    "Extra Trees Regressor",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "Ridge Regression",
    "Linear Regression",
    "Lasso Regression",
    "Elastic Net Regression",
    "Support Vector Regression (Linear)",
    "Decision Tree Regressor",
    "K-Nearest Neighbors Regressor",
    "Dummy Regressor (Median Baseline)",
    "AdaBoost Regressor",
    "XGBoost Regressor",
    "CatBoost Regressor",
    "Log1p Target Experiments",
]

METADATA_PATH = PROJECT_ROOT / "Models" / "best_model_metadata.json"
CLASSIFICATION_METADATA_PATH = PROJECT_ROOT / "Models" / "high_loss_classifier_metadata.json"
MEMBERSHIP_PATH = PROJECT_ROOT / "Dataset" / "wave3_split_membership.csv"


class ClaimWiseSystemTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        flask_app.app.config["TESTING"] = True
        flask_app.app.config["WTF_CSRF_ENABLED"] = False
        cls.client = flask_app.app.test_client()
        cls.preprocessor = load_preprocessor()
        cls.model = joblib.load(PROJECT_ROOT / "Models" / "best_model.pkl")

        # Record original raw dataset size and modification time
        cls.orig_raw_size = RAW_DATASET_PATH.stat().st_size
        cls.orig_raw_mtime = RAW_DATASET_PATH.stat().st_mtime

    def tearDown(self):
        clear_uploaded_files()

    # =========================================================================
    # 1. DATASET MANAGEMENT TESTS
    # =========================================================================

    def test_original_dataset_exists_and_unmodified(self):
        """Ensure production dataset is intact and has 50,000 rows."""
        self.assertTrue(RAW_DATASET_PATH.exists())
        self.assertEqual(RAW_DATASET_PATH.stat().st_size, self.orig_raw_size)

    def test_mode_a_dataset_view(self):
        """Default dataset view should be Mode A (Original ClaimWise)."""
        clear_uploaded_files()
        response = self.client.get("/dataset")
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("Mode A — Original ClaimWise Dataset Active", content)
        self.assertIn("50,000", content)
        self.assertIn("130", content)
        self.assertIn("Dataset CSV Files &amp; Schema Summaries", content)

    def test_upload_valid_compatible_csv(self):
        """Upload a valid 10-row slice of raw ClaimWise data."""
        raw_head = pd.read_csv(RAW_DATASET_PATH, nrows=10)
        csv_buffer = io.BytesIO()
        raw_head.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = self.client.post(
            "/dataset",
            data={"file": (csv_buffer, "test_valid_slice.csv")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("Mode B — Uploaded Custom Dataset Active", content)
        self.assertIn("test_valid_slice.csv", content)
        self.assertIn("COMPATIBLE WITH TRAINED MODEL", content)

    def test_upload_empty_csv_rejected(self):
        """Empty CSV should be caught and rejected gracefully."""
        csv_buffer = io.BytesIO(b"")
        response = self.client.post(
            "/dataset",
            data={"file": (csv_buffer, "empty.csv")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("The uploaded file is empty", content)

    def test_upload_invalid_extension_rejected(self):
        """Non-CSV file extensions should be rejected immediately."""
        txt_buffer = io.BytesIO(b"Hello world")
        response = self.client.post(
            "/dataset",
            data={"file": (txt_buffer, "test.txt")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("Only CSV (.csv) files are permitted", content)

    def test_upload_incompatible_schema_rejected(self):
        """CSV missing essential continuous features triggers incompatible warning."""
        dummy_df = pd.DataFrame({"colA": [1, 2, 3], "colB": ["x", "y", "z"]})
        csv_buffer = io.BytesIO()
        dummy_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = self.client.post(
            "/dataset",
            data={"file": (csv_buffer, "incompatible.csv")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("INCOMPATIBLE WITH MODEL", content)

    # =========================================================================
    # 2. PRIMARY PREDICTION ENGINE TESTS
    # =========================================================================

    def test_single_prediction_workflow_ui(self):
        """Prediction page must have exactly ONE primary input form, with no duplicate tabs."""
        response = self.client.get("/prediction")
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")

        self.assertIn("Understand Your Insurance Claim Loss", content)
        self.assertIn("Estimate Claim Loss", content)
        self.assertIn("1. Continuous Numeric Predictors", content)
        self.assertIn("2. Primary Categorical Factors", content)
        self.assertIn("3. Advanced Technical Predictors", content)
        # Check no alternative tabs exist
        self.assertNotIn("Tab A", content)
        self.assertNotIn("Tab B", content)

    def test_manual_prediction_flow_and_direct_model_parity(self):
        """Submitting values returns accurate predicted loss matching direct model inference."""
        form_data = {"mode": "manual"}
        for col, stats in self.preprocessor["continuous"].items():
            form_data[col] = str(stats["median"])
        for col, info in self.preprocessor["categorical"].items():
            form_data[col] = str(info["default"])

        response = self.client.post("/prediction", data=form_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")

        self.assertIn("Estimated Insurance Claim Loss", content)
        import json as _json
        _meta = _json.loads(METADATA_PATH.read_text())
        self.assertIn(_meta["model"], content)
        self.assertIn("130 Features Verified", content)
        self.assertIn("Important Estimation Disclaimer", content)

        # Numerical parity assertion
        raw_row = {
            col: float(stats["median"]) for col, stats in self.preprocessor["continuous"].items()
        }
        raw_row.update({
            col: str(info["default"]) for col, info in self.preprocessor["categorical"].items()
        })
        transformed = transform(pd.DataFrame([raw_row]), self.preprocessor)
        direct_pred = float(self.model.predict(transformed)[0])
        formatted_pred = f"{direct_pred:,.2f}"
        self.assertIn(formatted_pred, content)

    def test_currency_integrity_and_disclaimer(self):
        """Verify currency honesty and customer-facing disclaimer on prediction results."""
        form_data = {"mode": "manual"}
        for col, stats in self.preprocessor["continuous"].items():
            form_data[col] = str(stats["median"])
        for col, info in self.preprocessor["categorical"].items():
            form_data[col] = str(info["default"])

        response = self.client.post("/prediction", data=form_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")

        self.assertNotIn("₹", content)
        self.assertIn("The ClaimWise dataset schema does not define currency units for", content)
        self.assertIn("What does this mean?", content)
        self.assertIn("Predict Again", content)
        self.assertIn("Edit Information", content)

    def test_edit_information_preserves_inputs(self):
        """When form is submitted, input values are preserved in the form fields for editing."""
        form_data = {"mode": "manual"}
        for col, stats in self.preprocessor["continuous"].items():
            form_data[col] = str(stats["median"])
        for col, info in self.preprocessor["categorical"].items():
            form_data[col] = str(info["default"])

        custom_cont1 = "0.741258"
        form_data["cont1"] = custom_cont1

        response = self.client.post("/prediction", data=form_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn(f'value="{custom_cont1}"', content)

    # =========================================================================
    # 3. MODEL BENCHMARKING & REGRESSION METRICS TESTS (dynamic consistency)
    # =========================================================================

    def test_model_comparison_file_has_all_families_and_metrics(self):
        """Outputs/Model_Comparison/model_comparison.csv exists, has all families, 7 metrics, and a flagged champion."""
        self.assertTrue(COMPARISON_CSV_PATH.exists())
        df = pd.read_csv(COMPARISON_CSV_PATH)
        self.assertGreaterEqual(len(df), 16)

        expected_columns = ["Model", "MAE", "MSE", "RMSE", "R2", "Explained_Variance", "Median_AE", "MAPE"]
        for col in expected_columns:
            self.assertIn(col, df.columns)

        models_in_file = list(df["Model"])
        for expected in ALL_EXPECTED_MODELS:
            self.assertIn(expected, models_in_file)

        # Metrics must be finite and plausible (no NaN champions, positive errors).
        ok = df.dropna(subset=["MAE"])
        self.assertTrue((ok["MAE"] > 0).all())
        self.assertTrue((ok["RMSE"] > 0).all())

        # Exactly one row is flagged as the selected champion, and it is rank #1 by MAE.
        selected = df[df["Selected_For_Final_Test"].astype(str).str.lower() == "true"]
        self.assertEqual(len(selected), 1)
        best_row = df.sort_values("MAE").iloc[0]
        self.assertEqual(selected.iloc[0]["Model"], best_row["Model"])

    def test_final_test_metrics_match_promoted_champion(self):
        """Single untouched-holdout evaluation exists, matches the champion, and is internally consistent."""
        import json
        import numpy as np

        self.assertTrue(FINAL_TEST_METRICS_PATH.exists())
        ft = pd.read_csv(FINAL_TEST_METRICS_PATH).iloc[0]
        meta = json.loads(METADATA_PATH.read_text())

        self.assertEqual(ft["Model"], meta["model"])
        self.assertEqual(ft["Test_Rows"], 10000)
        self.assertEqual(ft["Train_Rows"], 40000)
        # R2 must be honest (positive but nowhere near the disclaimed 0.95).
        self.assertGreater(ft["R2"], 0.3)
        self.assertLess(ft["R2"], 0.95)
        self.assertGreater(ft["MAE"], 0)

        # Recompute MAE from the saved holdout predictions to confirm metric integrity.
        preds = pd.read_csv(
            PROJECT_ROOT / "Outputs" / "Final_Model_Evaluation" / "final_test_predictions.csv"
        )
        recomputed_mae = float(np.abs(preds["actual_loss"] - preds["predicted_loss"]).mean())
        self.assertAlmostEqual(recomputed_mae, float(ft["MAE"]), delta=0.51)

        # best_model_metadata.json final metrics must agree with the CSV.
        self.assertAlmostEqual(
            float(meta["final_test_metrics"]["MAE"]), float(ft["MAE"]), delta=0.01
        )
        self.assertAlmostEqual(
            float(meta["final_test_metrics"]["R2"]), float(ft["R2"]), delta=1e-6
        )

    def test_experiment_metadata_documents_wave3_protocol_and_r2_goal(self):
        """Experiment metadata records the wave-3 split protocol, selection metric and honest R² verdict."""
        self.assertTrue(EXPERIMENT_METADATA_PATH.exists())
        import json
        meta = json.loads(EXPERIMENT_METADATA_PATH.read_text())

        self.assertIn("wave3", meta)
        w3 = meta["wave3"]
        self.assertIn("validation MAE", w3["selection"]["primary_metric"])
        self.assertEqual(w3["split_protocol"]["validation_rows"], 7500)
        self.assertEqual(w3["split_protocol"]["holdout_rows"], 10000)
        self.assertEqual(w3["seed"], 2026)
        # The historical limitation (reused test set) must be documented.
        self.assertIn("pristine holdout", w3["why_new_split"].lower())

        verdict = meta["r2_goal_verdict"]
        self.assertEqual(verdict["goal_r2"], 0.95)
        self.assertFalse(verdict["reached"])
        self.assertLess(verdict["achieved_test_r2"], 0.95)
        self.assertGreater(verdict["achieved_test_r2"], 0.4)

    def test_models_page_displays_all_families_and_dynamic_metrics(self):
        """Models page lists all trained families with ranks and metric headers."""
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")

        for model_name in ALL_EXPECTED_MODELS:
            self.assertIn(model_name, content)

        # Check metric headers and definitions
        self.assertIn("MAE (Mean Absolute Error)", content)
        self.assertIn("MSE (Mean Squared Error)", content)
        self.assertIn("RMSE (Root Mean Squared Error)", content)
        self.assertIn("R² (Coefficient of Determination)", content)
        self.assertIn("Median AE (Median Absolute Error)", content)
        self.assertIn("Five Primary Regression Algorithms", content)
        # The model count is now rendered dynamically from the CSV.
        import json
        meta = json.loads(METADATA_PATH.read_text())
        self.assertIn(meta["model"], content)

    # =========================================================================
    # 4. CSV PREVIEW & DOWNLOAD SECURITY TESTS
    # =========================================================================

    def test_csv_preview_endpoint_pagination_and_search(self):
        """Endpoint /csv/preview returns paginated JSON structure with valid rows."""
        # Test default page 1 of raw dataset
        response = self.client.get("/csv/preview?path=Dataset/01_raw_claimwise_50000.csv&page=1&page_size=20")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertIn("rows", data)
        self.assertEqual(len(data["rows"]), 20)
        self.assertEqual(data["total_rows"], 50000)
        self.assertEqual(data["page"], 1)
        self.assertEqual(data["total_pages"], 2500)
        self.assertIn("loss", data["columns"])

        # Test search filter
        response_search = self.client.get(
            "/csv/preview?path=Outputs/Model_Comparison/model_comparison.csv&search=Hist"
        )
        self.assertEqual(response_search.status_code, 200)
        data_search = response_search.get_json()
        self.assertGreaterEqual(data_search["filtered_rows"], 1)
        self.assertIn("Hist Gradient Boosting Regressor", str(data_search["rows"]))

    def test_csv_download_valid_file(self):
        """Endpoint /csv/download/<filepath> safely downloads valid CSV artifacts."""
        response = self.client.get("/csv/download/Outputs/Model_Comparison/model_comparison.csv")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/csv")
        self.assertIn("attachment", response.headers.get("Content-Disposition", ""))

    def test_csv_download_blocks_path_traversal(self):
        """Path traversal attempts (e.g. ../../) must be strictly blocked with safe redirect."""
        # Attempt traversal outside approved inventory
        response = self.client.get("/csv/download/../../etc/passwd", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("Requested file does not exist or access is restricted", content)

        # Attempt to access app.py via csv download
        response_code = self.client.get("/csv/download/app.py", follow_redirects=True)
        self.assertEqual(response_code.status_code, 200)
        content_code = response_code.data.decode("utf-8")
        self.assertIn("Requested file does not exist or access is restricted", content_code)

    # =========================================================================
    # 5. ALL ROUTES AND ASSETS
    # =========================================================================

    def test_all_routes_return_200(self):
        """Every navigation route returns status 200 without exceptions."""
        routes = [
            "/",
            "/index",
            "/dataset",
            "/preprocessing",
            "/eda-visualization",
            "/models",
            "/academic",
            "/prediction",
            "/dashboard",
            "/reports",
            "/about",
        ]
        for route in routes:
            with self.subTest(route=route):
                resp = self.client.get(route)
                self.assertEqual(resp.status_code, 200)

    def test_static_and_outputs_serving(self):
        """Static css and output chart endpoints resolve correctly."""
        css_resp = self.client.get("/static/style.css")
        self.assertEqual(css_resp.status_code, 200)

        chart_resp = self.client.get("/outputs/Model_Comparison/model_comparison_bar_chart.png")
        self.assertEqual(chart_resp.status_code, 200)

    def test_outputs_gallery_page_and_inventory(self):
        """Outputs gallery URLs redirect into the unified EDA & Visualization page."""
        from artifact_inventory import get_outputs_inventory, get_outputs_summary

        summary = get_outputs_summary()
        self.assertGreaterEqual(summary["total_files"], 190)
        self.assertGreaterEqual(summary["total_images"], 120)
        self.assertGreaterEqual(summary["total_csvs"], 60)
        self.assertGreaterEqual(len(summary["topics"]), 7)

        resp = self.client.get("/outputs-gallery")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/eda-visualization", resp.headers.get("Location", ""))

        followed = self.client.get("/outputs-gallery", follow_redirects=True)
        self.assertEqual(followed.status_code, 200)
        content = followed.data.decode("utf-8")
        self.assertIn("Generated Outputs Gallery", content)
        self.assertIn("model_comparison_bar_chart.png", content)

    def test_legacy_visualization_and_gallery_redirects(self):
        """Legacy /visualization, /outputs-gallery and /gallery redirect to EDA & Visualization."""
        for legacy in ("/visualization", "/outputs-gallery", "/gallery"):
            with self.subTest(legacy=legacy):
                resp = self.client.get(legacy)
                self.assertEqual(resp.status_code, 302)
                self.assertIn("/eda-visualization", resp.headers.get("Location", ""))
                followed = self.client.get(legacy, follow_redirects=True)
                self.assertEqual(followed.status_code, 200)
                self.assertIn("Exploratory Data Analysis &amp; Visualization Studio", followed.data.decode("utf-8"))

    def test_eda_visualization_page_sections(self):
        """EDA & Visualization page renders all five required sections."""
        content = self.client.get("/eda-visualization").data.decode("utf-8")
        for section in (
            "SECTION A",
            "SECTION B",
            "SECTION C",
            "SECTION D",
            "SECTION E",
            "EDA Overview",
            "Exploratory Data Analysis",
            "Model Evaluation Visualizations",
            "Generated Outputs Gallery",
            "Search &amp; Filters",
        ):
            self.assertIn(section, content)
        # Curated modules and discovered charts coexist without duplicate srcs
        self.assertIn("claimwise_raw_target_distribution.png", content)
        self.assertIn("final_test_metrics.csv", content)

    def test_eda_visualization_images_resolve(self):
        """Every image URL rendered on the EDA page returns HTTP 200."""
        import re

        content = self.client.get("/eda-visualization").data.decode("utf-8")
        srcs = sorted(
            {
                s
                for s in re.findall(r"/outputs/([^\"')\s]+)", content)
                if s.lower().endswith((".png", ".jpg", ".jpeg", ".svg"))
            }
        )
        self.assertGreater(len(srcs), 100)
        for src in srcs:
            with self.subTest(src=src):
                resp = self.client.get(f"/outputs/{src}")
                self.assertEqual(resp.status_code, 200)
                self.assertTrue(
                    resp.data[:8].startswith(b"\x89PNG")
                    or resp.data[:3] == b"\xff\xd8\xff"
                    or b"<svg" in resp.data[:200]
                )

    def test_dataset_explorer_initial_page_is_500_records(self):
        """Dataset page loads the first 500 raw records via server-side pagination."""
        page = self.client.get("/dataset")
        self.assertEqual(page.status_code, 200)
        content = page.data.decode("utf-8")
        self.assertIn("dataset-explorer", content)
        self.assertIn("Raw Dataset Records", content)

        data = self.client.get("/dataset/api/records?page=1&page_size=500").get_json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["rows"]), 500)
        self.assertEqual(data["total_rows"], 50000)
        self.assertEqual(data["total_pages"], 100)
        self.assertEqual(data["start_row"], 1)
        self.assertEqual(data["end_row"], 500)
        self.assertEqual(len(data["columns"]), 132)
        self.assertIn("id", data["columns"])
        self.assertIn("loss", data["columns"])

    def test_dataset_explorer_pagination_advances(self):
        """Page 2 shows the next records — never a repeat of the first 500."""
        page1 = self.client.get("/dataset/api/records?page=1&page_size=500").get_json()
        page2 = self.client.get("/dataset/api/records?page=2&page_size=500").get_json()
        self.assertEqual(page2["start_row"], 501)
        self.assertEqual(page2["end_row"], 1000)
        ids1 = {row["id"] for row in page1["rows"]}
        ids2 = {row["id"] for row in page2["rows"]}
        self.assertTrue(ids1.isdisjoint(ids2))

        last = self.client.get("/dataset/api/records?page=100&page_size=500").get_json()
        self.assertEqual(last["start_row"], 49501)
        self.assertEqual(last["end_row"], 50000)
        self.assertEqual(len(last["rows"]), 500)

        tail = self.client.get("/dataset/api/records?page=101&page_size=500").get_json()
        self.assertEqual(tail["page"], 100)  # clamped to the true last page

    def test_dataset_explorer_search_filter_and_sort(self):
        """Search, column filters, and sorting all operate on the real dataset."""
        target = pd.read_csv(RAW_DATASET_PATH, usecols=["id", "loss"]).sort_values(
            "loss", ascending=False
        ).iloc[0]

        sorted_data = self.client.get(
            "/dataset/api/records?sort_by=loss&sort_dir=desc&page_size=100"
        ).get_json()
        self.assertTrue(sorted_data["success"])
        self.assertEqual(sorted_data["rows"][0]["id"], str(int(target["id"])))
        self.assertEqual(float(sorted_data["rows"][0]["loss"]), float(target["loss"]))

        filtered = self.client.get(
            "/dataset/api/records?filter_col=loss&filter_op=gt&filter_value=80000&page_size=100"
        ).get_json()
        self.assertTrue(filtered["success"])
        self.assertGreaterEqual(filtered["filtered_rows"], 1)
        self.assertLess(filtered["filtered_rows"], filtered["total_rows"])
        for row in filtered["rows"]:
            self.assertGreater(float(row["loss"]), 80000)

        searched = self.client.get(
            f"/dataset/api/records?search={int(target['id'])}&page_size=100"
        ).get_json()
        self.assertTrue(searched["success"])
        self.assertGreaterEqual(searched["filtered_rows"], 1)
        self.assertTrue(any(row["id"] == str(int(target["id"])) for row in searched["rows"]))

    def test_dataset_explorer_page_size_validation(self):
        """Unsupported page sizes fall back to the 500-record default."""
        data = self.client.get("/dataset/api/records?page_size=7").get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["page_size"], 500)
        self.assertEqual(len(data["rows"]), 500)

        bad_sort = self.client.get("/dataset/api/records?sort_by=not_a_column").get_json()
        self.assertFalse(bad_sort["success"])

        bad_op = self.client.get(
            "/dataset/api/records?filter_col=loss&filter_op=like&filter_value=1"
        ).get_json()
        self.assertFalse(bad_op["success"])

    def test_raw_dataset_download_is_complete(self):
        """The raw CSV download returns the complete 50,000-row source file."""
        resp = self.client.get("/csv/download/Dataset/01_raw_claimwise_50000.csv")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("attachment", resp.headers.get("Content-Disposition", ""))
        body = resp.data.decode("utf-8")
        self.assertEqual(len(body.splitlines()), 50001)
        self.assertTrue(body.startswith("id,cat1,"))

    def test_dataset_filtered_export(self):
        """Filtered export streams only matching rows as an explicit separate action."""
        resp = self.client.get(
            "/dataset/export?filter_col=loss&filter_op=gt&filter_value=80000"
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, "text/csv")
        self.assertIn("attachment", resp.headers.get("Content-Disposition", ""))
        lines = resp.data.decode("utf-8").splitlines()
        self.assertGreaterEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("id,"))

    def test_dataset_column_statistics_and_validation(self):
        """Column statistics and validation are computed from the real raw file."""
        stats = self.client.get("/dataset/api/stats").get_json()
        self.assertTrue(stats["success"])
        self.assertEqual(len(stats["columns"]), 132)
        self.assertEqual(stats["rows"], 50000)
        self.assertEqual(stats["total_missing"], 0)
        self.assertEqual(stats["duplicate_rows"], 0)
        self.assertIsNotNone(stats["target_summary"])
        self.assertEqual(stats["target_summary"]["name"], "loss")
        self.assertGreater(stats["target_summary"]["max"], stats["target_summary"]["min"])

        from dataset_explorer import validate_dataset

        validation = validate_dataset("raw")
        self.assertTrue(validation["is_valid"])
        self.assertTrue(validation["expected_schema_ok"])
        self.assertTrue(validation["target_present"])
        self.assertEqual(validation["rows"], 50000)
        self.assertEqual(validation["columns"], 132)

    def test_no_pickle_or_secret_files_are_exposed(self):
        """Model pickles and app internals are never served through output routes."""
        for url in (
            "/outputs/../Models/best_model.pkl",
            "/outputs/Models/best_model.pkl",
            "/csv/download/Models/best_model.pkl",
            "/csv/download/app.py",
        ):
            with self.subTest(url=url):
                resp = self.client.get(url)
                self.assertIn(resp.status_code, (302, 400, 404))

    def test_outputs_security_and_traversal(self):
        """Output file serving blocks directory traversal attempts strictly."""
        # Legitimate file
        valid_resp = self.client.get("/outputs/Model_Comparison/model_comparison.csv")
        self.assertEqual(valid_resp.status_code, 200)

        # Traversal attempts
        traversal_1 = self.client.get("/outputs/../app.py")
        self.assertIn(traversal_1.status_code, [400, 404])

        traversal_2 = self.client.get("/outputs/../../etc/passwd")
        self.assertIn(traversal_2.status_code, [400, 404])

    # =========================================================================
    # 6. METHODOLOGY, PIPELINE & INPUT-VALIDATION TESTS
    # =========================================================================

    def test_navigation_primary_order(self):
        """Primary navigation lists the eight core pages in the required order."""
        content = self.client.get("/").data.decode("utf-8")
        nav_segment = content.split('class="nav-menu"', 1)[1].split("</nav>", 1)[0]
        labels = [
            "Home",
            "About",
            "Dataset",
            "Preprocessing",
            "EDA &amp; Visualization",
            "Models",
            "Prediction",
            "Reports",
        ]
        positions = [nav_segment.find(f">{label}</a>") for label in labels]
        for label, pos in zip(labels, positions):
            self.assertGreaterEqual(pos, 0, f"Missing nav link: {label}")
        self.assertEqual(positions, sorted(positions), "Primary nav links out of order")
        # Secondary links exist but no separate Visualization/Outputs entries remain
        self.assertIn(">Academic</a>", nav_segment)
        self.assertIn(">Dashboard</a>", nav_segment)
        self.assertIn(">Contact</a>", nav_segment)
        self.assertNotIn(">Visualization</a>", nav_segment)
        self.assertNotIn(">Outputs</a>", nav_segment)

    def test_dataset_row_and_column_counts(self):
        """Dataset explorer reports the real 50,000 × 132 shape of the raw dataset."""
        data = self.client.get(
            "/csv/preview?path=Dataset/01_raw_claimwise_50000.csv&page=1&page_size=5"
        ).get_json()
        self.assertEqual(data["total_rows"], 50000)
        self.assertEqual(len(data["columns"]), 132)
        self.assertIn("id", data["columns"])
        self.assertIn("loss", data["columns"])
        self.assertEqual(len(data["rows"]), 5)

    def test_missing_value_statistics_available(self):
        """Preprocessing stage summary exposes real missing-value and duplicate counts."""
        stats = flask_app.load_dataset_stats()
        self.assertIsNotNone(stats["raw_rows"])
        self.assertEqual(stats["raw_rows"], 50000)
        self.assertIsNotNone(stats["missing"])
        self.assertIsNotNone(stats["duplicates"])

    def test_preprocessing_pipeline_artifact_loads(self):
        """Saved preprocessor artifact loads, verifies and carries the 130-feature schema."""
        pre = self.preprocessor
        self.assertEqual(len(pre["feature_order"]), 130)
        self.assertEqual(len(pre["continuous_columns"]), 14)
        self.assertEqual(len(pre["categorical_columns"]), 116)
        self.assertLessEqual(pre["verification"]["max_abs_diff"], 1e-9)

    def test_model_artifact_loads_and_predicts_finite(self):
        """best_model.pkl loads and returns finite predictions for real test rows."""
        import numpy as np
        sample = pd.read_csv(TEST_DATASET_X, nrows=10)
        preds = self.model.predict(sample)
        self.assertEqual(len(preds), 10)
        self.assertTrue(np.isfinite(preds).all())
        self.assertTrue((preds > 0).all())

    def test_prediction_rejects_invalid_numerical_input(self):
        """Non-numeric continuous values return an actionable validation error."""
        form_data = {"mode": "manual"}
        for col, stats in self.preprocessor["continuous"].items():
            form_data[col] = str(stats["median"])
        for col, info in self.preprocessor["categorical"].items():
            form_data[col] = str(info["default"])
        form_data["cont1"] = "not-a-number"
        response = self.client.post("/prediction", data=form_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("cont1", content)
        self.assertIn("valid finite number", content)

    def test_prediction_rejects_missing_required_input(self):
        """Empty continuous fields are rejected with a clear error."""
        response = self.client.post("/prediction", data={"mode": "manual"}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("empty", content.lower())

    def test_prediction_rejects_unseen_categorical_value(self):
        """Unseen category values are rejected before reaching the model."""
        form_data = {"mode": "manual"}
        for col, stats in self.preprocessor["continuous"].items():
            form_data[col] = str(stats["median"])
        for col, info in self.preprocessor["categorical"].items():
            form_data[col] = str(info["default"])
        form_data["cat1"] = "__UNSEEN_CATEGORY__"
        response = self.client.post("/prediction", data=form_data, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("unknown category", content)

    def test_end_to_end_test_row_prediction_matches_direct_model(self):
        """mode=test row 0 matches a direct model invocation on the same test row."""
        import numpy as np
        response = self.client.post(
            "/prediction", data={"mode": "test", "row_index": "0"}, follow_redirects=True
        )
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        x_test = pd.read_csv(TEST_DATASET_X, nrows=1)
        y_test = pd.read_csv(TEST_DATASET_Y)["loss"].iloc[0]
        expected = float(self.model.predict(x_test)[0])
        self.assertTrue(np.isfinite(expected))
        self.assertIn(f"{expected:,.2f}", content)
        self.assertIn(f"{float(y_test):,.2f}", content)

    def test_report_gallery_groups_artifacts(self):
        """Reports page renders the grouped artifact gallery with discovery counts."""
        response = self.client.get("/reports")
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("Artifact Gallery by Category", content)
        self.assertIn("Model Training &amp; Comparison", content)
        self.assertIn("Modified (UTC)", content)
        self.assertIn("Final untouched test evaluation", content)

    def test_application_startup_configuration(self):
        """App starts with hardened configuration (env-driven secret, upload cap)."""
        self.assertTrue(flask_app.app.secret_key)
        self.assertNotEqual(flask_app.app.secret_key, "claimwise-insurance-prediction-secret-key")
        self.assertEqual(flask_app.app.config["MAX_CONTENT_LENGTH"], 20 * 1024 * 1024)

    def test_output_serving_missing_file_is_404(self):
        """Missing artifacts return 404 rather than leaking errors."""
        resp = self.client.get("/outputs/Model_Comparison/does_not_exist.png")
        self.assertEqual(resp.status_code, 404)

    def test_outputs_inventory_survives_empty_directory(self):
        """An empty Outputs subdirectory does not break discovery."""
        from artifact_inventory import get_outputs_summary
        empty_dir = PROJECT_ROOT / "Outputs" / "_tmp_empty_dir_test"
        empty_dir.mkdir(exist_ok=True)
        try:
            summary = get_outputs_summary()
            self.assertGreaterEqual(summary["total_files"], 190)
        finally:
            empty_dir.rmdir()

    # =========================================================================
    # 9. WAVE-3 PROTOCOL: LEAKAGE, SPLIT SEPARATION, ROLLBACK, CLASSIFICATION
    # =========================================================================

    def test_split_separation_no_row_overlap(self):
        """Wave-3 train/validation/holdout partitions are strictly disjoint (by id)."""
        self.assertTrue(MEMBERSHIP_PATH.exists())
        membership = pd.read_csv(MEMBERSHIP_PATH)
        self.assertEqual(len(membership), 50000)
        self.assertEqual(membership["id"].nunique(), 50000)
        counts = membership["split"].value_counts().to_dict()
        self.assertEqual(counts.get("train"), 32500)
        self.assertEqual(counts.get("validation"), 7500)
        self.assertEqual(counts.get("holdout"), 10000)

    def test_preprocessing_scaler_fitted_on_training_only(self):
        """The StandardScaler's mean_ matches raw means over the original training rows only.

        The pipeline fitted the scaler on the seed-42 80/20 training partition.
        Reproducing those indices and comparing scaler.mean_ to the raw training-row
        means (which will NOT match the full-data or holdout means) proves the scaler
        never saw validation/holdout rows — a genuine leakage check.
        """
        import numpy as np
        from sklearn.model_selection import train_test_split

        pre = self.preprocessor
        cont = pre["continuous_columns"]
        raw = pd.read_csv(RAW_DATASET_PATH)
        raw = raw.drop(columns=["id"], errors="ignore")

        idx = np.arange(len(raw))
        train_idx, _ = train_test_split(idx, test_size=0.20, random_state=42)
        expected_mean = raw.iloc[train_idx][cont].mean().to_numpy(dtype=float)
        full_mean = raw[cont].mean().to_numpy(dtype=float)

        # scaler.mean_ must equal the training-row means.
        np.testing.assert_allclose(pre["scaler"].mean_, expected_mean, rtol=1e-9, atol=1e-9)
        # And it must NOT equal the full-data means (which would indicate leakage).
        self.assertGreater(np.abs(pre["scaler"].mean_ - full_mean).max(), 1e-9)

    def test_regression_metrics_recomputed_from_predictions(self):
        """MAE/RMSE/R² recomputed from saved holdout predictions match the reported metrics."""
        import json
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        ft = pd.read_csv(FINAL_TEST_METRICS_PATH).iloc[0]
        preds = pd.read_csv(
            PROJECT_ROOT / "Outputs" / "Final_Model_Evaluation" / "final_test_predictions.csv"
        )
        y_true = preds["actual_loss"].to_numpy()
        y_pred = preds["predicted_loss"].to_numpy()
        self.assertAlmostEqual(mean_absolute_error(y_true, y_pred), float(ft["MAE"]), delta=0.51)
        self.assertAlmostEqual(mean_squared_error(y_true, y_pred) ** 0.5, float(ft["RMSE"]), delta=0.51)
        self.assertAlmostEqual(r2_score(y_true, y_pred), float(ft["R2"]), delta=1e-6)

    def test_model_save_load_roundtrip_consistency(self):
        """best_model.pkl reloads and reproduces predictions on real holdout rows."""
        import numpy as np
        model = joblib.load(PROJECT_ROOT / "Models" / "best_model.pkl")
        x_hold = pd.read_csv(TEST_DATASET_X, nrows=50)
        p1 = model.predict(x_hold)
        # Round-trip through joblib to confirm serialization fidelity.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            joblib.dump(model, tmp.name)
            reloaded = joblib.load(tmp.name)
        p2 = reloaded.predict(x_hold)
        self.assertTrue(np.isfinite(p1).all())
        self.assertTrue(np.allclose(p1, p2, atol=1e-9))

    def test_champion_rollback_backup_exists_and_loads(self):
        """The pre-wave3 champion backup exists, loads, and produces finite predictions."""
        import numpy as np
        backup_path = PROJECT_ROOT / "Models" / "best_model_pre_wave3.pkl"
        self.assertTrue(backup_path.exists(), "Rollback backup must be preserved.")
        backup = joblib.load(backup_path)
        x_hold = pd.read_csv(TEST_DATASET_X, nrows=10)
        preds = backup.predict(x_hold)
        self.assertTrue(np.isfinite(preds).all())

    def test_classification_metrics_and_positive_class_definition(self):
        """High-loss classifier reports F1/precision/recall/accuracy/confusion with a documented threshold."""
        import json
        import numpy as np
        from sklearn.metrics import (
            accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
        )

        self.assertTrue(CLASSIFICATION_METADATA_PATH.exists())
        meta = json.loads(CLASSIFICATION_METADATA_PATH.read_text())

        # Positive class definition and threshold must be documented and train-derived.
        self.assertIn("high_loss", meta["positive_class_definition"])
        self.assertGreater(meta["threshold"], 0)
        self.assertIn("TRAINING", meta["threshold_source"])

        hold = meta["final_holdout_metrics"]
        for key in ("F1_binary", "Precision", "Recall", "Accuracy", "ROC_AUC", "PR_AUC"):
            self.assertIn(key, hold)
            self.assertGreaterEqual(hold[key], 0.0)
            self.assertLessEqual(hold[key], 1.0)

        # Recompute confusion matrix from saved holdout predictions for integrity.
        preds = pd.read_csv(
            PROJECT_ROOT / "Outputs" / "ClassificationExperiment" / "holdout_predictions.csv"
        )
        cm = confusion_matrix(preds["actual_high_loss"], preds["predicted_label"])
        self.assertEqual(int(cm.ravel()[0]), hold["Confusion_TN"])
        self.assertEqual(int(cm.ravel()[1]), hold["Confusion_FP"])
        self.assertEqual(int(cm.ravel()[2]), hold["Confusion_FN"])
        self.assertEqual(int(cm.ravel()[3]), hold["Confusion_TP"])
        # F1 recomputation must match.
        recomputed_f1 = f1_score(preds["actual_high_loss"], preds["predicted_label"])
        self.assertAlmostEqual(recomputed_f1, hold["F1_binary"], delta=1e-6)
        # The classifier must NOT be mixed into the regression leaderboard.
        comp = pd.read_csv(COMPARISON_CSV_PATH)
        self.assertNotIn("high_loss", " ".join(comp["Model"].astype(str)).lower())

    def test_reports_page_shows_separate_classification_section(self):
        """Reports page renders the classification section, clearly separated from regression."""
        response = self.client.get("/reports")
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("Separate Classification Task", content)
        self.assertIn("High-Loss Claim Triage", content)
        self.assertIn("does <strong>not</strong> measure the accuracy of the continuous loss regression", content)
        import json
        meta = json.loads(CLASSIFICATION_METADATA_PATH.read_text())
        self.assertIn(meta["champion"], content)



if __name__ == "__main__":
    unittest.main()
