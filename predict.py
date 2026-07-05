import sys
import os
import joblib
import pandas as pd

def load_assets():
    model_path = "best_model.pkl"
    scaler_path = "scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Error: Saved model assets ('best_model.pkl' and 'scaler.pkl') were not found.")
        print("Please run 'python train.py' first to train and save the model.")
        sys.exit(1)
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_sales(tv, radio, newspaper, model, scaler):
    # Form input DataFrame (matching training features exactly)
    input_data = pd.DataFrame({
        "TV": [tv],
        "Radio": [radio],
        "Newspaper": [newspaper]
    })
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    return prediction

def main():
    print("====================================================")
    print("     Sales Prediction Inference Interface (ML)      ")
    print("====================================================")
    
    model, scaler = load_assets()
    
    # Check if arguments are passed via CLI
    if len(sys.argv) == 4:
        try:
            tv = float(sys.argv[1])
            radio = float(sys.argv[2])
            newspaper = float(sys.argv[3])
        except ValueError:
            print("Error: All inputs (TV, Radio, Newspaper spends) must be numbers.")
            sys.exit(1)
    else:
        print("Enter the planned marketing spend ($) for each platform below:")
        try:
            tv = float(input("  TV Advertising Spend ($): "))
            radio = float(input("  Radio Advertising Spend ($): "))
            newspaper = float(input("  Newspaper Advertising Spend ($): "))
        except ValueError:
            print("Error: Please enter valid numerical values.")
            sys.exit(1)
            
    # Perform prediction
    predicted_sales = predict_sales(tv, radio, newspaper, model, scaler)
    
    print("\n----------------------------------------------------")
    print(f"Results of Analysis:")
    print(f"  TV Spend:         ${tv:,.2f}")
    print(f"  Radio Spend:      ${radio:,.2f}")
    print(f"  Newspaper Spend:  ${newspaper:,.2f}")
    print(f"  --> Predicted Sales: ${predicted_sales:,.2f}")
    print("----------------------------------------------------")

if __name__ == "__main__":
    main()
