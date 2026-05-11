from src.predict import predict, predict_with_explainability, CLASS_NAMES


def test_predict_returns_required_keys(model, device, sample_image):
    result = predict(sample_image, model, device)
    assert {"predicted_class", "class_name", "description", "confidence", "probabilities"} <= result.keys()


def test_predicted_class_in_valid_range(model, device, sample_image):
    result = predict(sample_image, model, device)
    assert 0 <= result["predicted_class"] <= 4


def test_confidence_is_valid_probability(model, device, sample_image):
    result = predict(sample_image, model, device)
    assert 0.0 <= result["confidence"] <= 1.0


def test_probabilities_sum_to_one(model, device, sample_image):
    result = predict(sample_image, model, device)
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 1e-3


def test_class_name_matches_mapping(model, device, sample_image):
    result = predict(sample_image, model, device)
    assert result["class_name"] == CLASS_NAMES[result["predicted_class"]]


def test_five_probability_entries(model, device, sample_image):
    result = predict(sample_image, model, device)
    assert len(result["probabilities"]) == 5


def test_predict_with_explainability_includes_heatmap(model, device, sample_image):
    result = predict_with_explainability(sample_image, model, device)
    assert "gradcam_heatmap" in result
    assert isinstance(result["gradcam_heatmap"], str)
    assert len(result["gradcam_heatmap"]) > 100
