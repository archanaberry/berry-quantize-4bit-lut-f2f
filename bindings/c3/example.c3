module example;

import berrybits;

fn int main() {
    float[8] teacher_weights = {0.1, -0.2, 0.3, -0.4, -0.4, 0.3, -0.2, 0.1};
    float[8] calibration_inputs = {1.0, 0.5, -0.5, 0.25, -0.25, 0.75, 1.0, -1.0};
    float[2] output;
    float rmse_before;
    float rmse_after;

    void* model = berrybits_model_create(2, 4);
    if (model == null) return 1;

    if (!berrybits_model_distill_linear(model, &teacher_weights[0], null,
            &calibration_inputs[0], 2, 4, 0.02,
            &rmse_before, &rmse_after)) {
        berrybits_model_destroy(model);
        return 2;
    }
    if (!berrybits_model_gemv(model, &calibration_inputs[0], &output[0])) {
        berrybits_model_destroy(model);
        return 3;
    }

    berrybits_model_save(model, "build/c3_api_test.b4lut");
    berrybits_model_destroy(model);
    return 0;
}
