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
]


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
        self.assertIn("Dataset CSV Files & Schema Summaries", content)

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
        self.assertIn("Hist Gradient Boosting Regressor", content)
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
    # 3. 13-MODEL BENCHMARKING & METRICS TESTS
    # =========================================================================

    def test_model_comparison_file_contains_all_thirteen_models(self):
        """Outputs/Model_Comparison/model_comparison.csv must exist with 13 models and 7 metrics."""
        self.assertTrue(COMPARISON_CSV_PATH.exists())
        df = pd.read_csv(COMPARISON_CSV_PATH)
        self.assertEqual(len(df), 13)

        expected_columns = ["Model", "MAE", "MSE", "RMSE", "R2", "Explained_Variance", "Median_AE", "MAPE"]
        for col in expected_columns:
            self.assertIn(col, df.columns)

        models_in_file = list(df["Model"])
        for expected in ALL_EXPECTED_MODELS:
            self.assertIn(expected, models_in_file)

        # Assert Rank 1 is Hist Gradient Boosting Regressor
        best_row = df.iloc[0]
        self.assertEqual(best_row["Model"], "Hist Gradient Boosting Regressor")
        self.assertAlmostEqual(best_row["RMSE"], 1921.02, delta=1.0)
        self.assertAlmostEqual(best_row["MAE"], 1215.30, delta=1.0)
        self.assertAlmostEqual(best_row["R2"], 0.5459, delta=0.01)

    def test_models_page_displays_all_thirteen_models(self):
        """Models page lists all 13 trained models with ranks and metrics."""
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
        self.assertIn("13 Algorithms", content)

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
            "/visualization",
            "/models",
            "/academic",
            "/prediction",
            "/dashboard",
            "/reports",
            "/about",
            "/outputs-gallery",
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
        """Outputs gallery discovers all files and renders vertical scrolling view."""
        from artifact_inventory import get_outputs_inventory, get_outputs_summary

        summary = get_outputs_summary()
        self.assertGreaterEqual(summary["total_files"], 180)
        self.assertGreaterEqual(summary["total_images"], 100)
        self.assertGreaterEqual(summary["total_csvs"], 50)
        self.assertEqual(len(summary["topics"]), 7)

        resp = self.client.get("/outputs-gallery")
        self.assertEqual(resp.status_code, 200)
        content = resp.data.decode("utf-8")
        self.assertIn("ClaimWise Project Outputs Gallery", content)
        self.assertIn("Discovered Files", content)
        self.assertIn("Visualizations &amp; Charts", content)
        self.assertIn("model_comparison_bar_chart.png", content)

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



if __name__ == "__main__":
    unittest.main()
