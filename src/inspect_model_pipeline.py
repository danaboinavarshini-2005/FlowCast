import joblib

models = {
    "Traffic Volume": "models/traffic_volume_model.pkl",
    "Congestion": "models/congestion_model.pkl",
    "Travel Time": "models/travel_time_model.pkl",
    "Future Accident V4": "models/future_accident_v4_model.pkl"
}

for name, path in models.items():
    print(f"\n========== {name} ==========")

    model = joblib.load(path)

    for step_name, step in model.named_steps.items():
        print(f"\nStep: {step_name}")
        print("Type:", type(step))

        if hasattr(step, "transformers_"):
            for transformer_name, transformer, columns in step.transformers_:
                print("\nTransformer:", transformer_name)
                print("Columns:", columns)

                if hasattr(transformer, "categories_"):
                    print("Categories:", transformer.categories_)