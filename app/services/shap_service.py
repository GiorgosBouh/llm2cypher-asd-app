from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from app.data.schema_validation import feature_dataframe_from_case
from app.services.tabular_model_service import ModelBundle
from app.utils.text_summaries import prediction_plain_language


@dataclass
class LocalExplanation:
    contributions: pd.DataFrame
    plain_language: str


class ShapService:
    """Explain the tabular predictor using real questionnaire and demographic features."""

    def _transformed_data(self, bundle: ModelBundle, features_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        preprocessor = bundle.pipeline.named_steps["preprocessor"]
        transformed = preprocessor.transform(features_df)
        feature_names = list(preprocessor.get_feature_names_out())
        return transformed, feature_names

    def _tree_explainer(self, bundle: ModelBundle):
        model = bundle.pipeline.named_steps["model"]
        background_transformed, _ = self._transformed_data(bundle, bundle.train_sample)
        return shap.TreeExplainer(model, background_transformed)

    def global_importance(self, bundle: ModelBundle) -> pd.DataFrame:
        background = bundle.train_sample.copy()
        transformed, feature_names = self._transformed_data(bundle, background)
        explainer = self._tree_explainer(bundle)
        shap_values = explainer.shap_values(transformed)
        values = shap_values[1] if isinstance(shap_values, list) else shap_values
        importance = np.abs(values).mean(axis=0)
        return (
            pd.DataFrame({"feature": feature_names, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def local_explanation(self, bundle: ModelBundle, case_data: dict[str, object], predicted_label: str, probability: float) -> LocalExplanation:
        case_df = feature_dataframe_from_case(case_data)
        transformed, feature_names = self._transformed_data(bundle, case_df)
        explainer = self._tree_explainer(bundle)
        shap_values = explainer.shap_values(transformed)
        values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        contributions = (
            pd.DataFrame({"feature": feature_names, "shap_value": values, "abs_value": np.abs(values)})
            .sort_values("abs_value", ascending=False)
            .reset_index(drop=True)
        )

        display_features = [feature.split("__", 1)[-1] for feature in contributions["feature"].head(5).tolist()]
        summary = prediction_plain_language(display_features, predicted_label, probability)
        return LocalExplanation(contributions=contributions, plain_language=summary)
