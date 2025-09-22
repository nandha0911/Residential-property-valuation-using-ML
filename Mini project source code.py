# Import necessary libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from tkinter import messagebox, filedialog
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize empty DataFrame and model components
df = pd.DataFrame()
label_encoder = LabelEncoder()
model = LinearRegression()

# Function to load dataset
def load_dataset():
    global df, label_encoder, model, location_var

    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return

    try:
        df = pd.read_csv(file_path)

        if not {'Rooms', 'SquareFoot', 'Location', 'Price'}.issubset(df.columns):
            messagebox.showerror("Invalid File", "CSV must contain 'Rooms', 'SquareFoot', 'Location', 'Price' columns.")
            return

        # Encode 'Location'
        label_encoder = LabelEncoder()
        df['Location_encoded'] = label_encoder.fit_transform(df['Location'])

        # Prepare features and target
        X = df[['Rooms', 'SquareFoot', 'Location_encoded']]
        y = df['Price']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Update available locations dropdown
        locations = list(label_encoder.classes_)
        location_var.set(locations[0])  # Set default selection

        # Clear old options
        location_menu['menu'].delete(0, 'end')
        for loc in locations:
            location_menu['menu'].add_command(label=loc, command=tk._setit(location_var, loc))

        # Update info label
        locations_info = f"Available Locations: {', '.join(locations)}"

        messagebox.showinfo("Success", "Dataset loaded and model trained successfully!")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset: {e}")

# Store last prediction values
last_prediction = {}

# Function to predict house price
def predict_price():
    global last_prediction
    try:
        rooms = int(rooms_entry.get())
        square_foot = int(sqft_entry.get())
        location = location_var.get()

        location_encoded = label_encoder.transform([location])[0]
        input_data = [[rooms, square_foot, location_encoded]]
        predicted_price = model.predict(input_data)[0]

        lower_price = round(predicted_price * 0.9, -3)
        upper_price = round(predicted_price * 1.1, -3)

        result_label.config(text=f"Predicted Price Range: ${lower_price:,.0f} to ${upper_price:,.0f}")

        # Save USD prediction details
        last_prediction = {
            'Rooms': rooms,
            'Square Footage': square_foot,
            'Location': location,
            'Lower USD': lower_price,
            'Upper USD': upper_price,
            'Lower INR': '',
            'Upper INR': ''
        }

    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values for rooms and square footage.")
    except Exception as e:
        messagebox.showerror("Error", str(e))



# Function to convert USD to INR
def convert_to_inr():
    global last_prediction
    try:
        predicted_price_text = result_label.cget("text")
        if "Predicted Price Range" not in predicted_price_text:
            messagebox.showerror("No Prediction", "Please predict the price first.")
            return

        price_range = predicted_price_text.replace("Predicted Price Range: $", "")
        lower_usd, upper_usd = price_range.split(" to $")
        lower_usd = float(lower_usd.replace(',', ''))
        upper_usd = float(upper_usd.replace(',', ''))

        usd_to_inr_rate = 83.0

        lower_inr = round(lower_usd * usd_to_inr_rate, -3)
        upper_inr = round(upper_usd * usd_to_inr_rate, -3)

        inr_result_label.config(text=f"Predicted Price Range in INR: ₹{lower_inr:,.0f} to ₹{upper_inr:,.0f}")

        # Save INR prediction details
        last_prediction['Lower INR'] = lower_inr
        last_prediction['Upper INR'] = upper_inr

    except Exception as e:
        messagebox.showerror("Error", f"Failed to convert price: {e}")


def save_prediction():
    if not last_prediction:
        messagebox.showerror("No Prediction", "Please predict a price before saving.")
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return

    try:
        # Check if file exists — append header only if new
        write_header = not os.path.isfile(file_path)

        import csv
        with open(file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=last_prediction.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(last_prediction)

        messagebox.showinfo("Success", "Prediction saved successfully!")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to save prediction: {e}")

def visualize_data():
    if df.empty:
        messagebox.showerror("No Data", "Please load the dataset first.")
        return

    # Plot 1: Distribution of House Prices
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Price'], kde=True, bins=30)
    plt.title('Distribution of House Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Plot 2: Average Price by Location
    plt.figure(figsize=(8, 5))
    avg_price_by_location = df.groupby('Location')['Price'].mean().reset_index()
    sns.barplot(data=avg_price_by_location, x='Location', y='Price')
    plt.title('Average House Price by Location')
    plt.xlabel('Location')
    plt.ylabel('Average Price')
    plt.grid(True)
    plt.show()

    # Plot 3: Square Footage vs. Price Scatterplot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='SquareFoot', y='Price', hue='Location')
    plt.title('Square Footage vs. Price')
    plt.xlabel('Square Footage')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

    # Plot 4: Rooms vs. Price Boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Rooms', y='Price')
    plt.title('House Price by Number of Rooms')
    plt.xlabel('Number of Rooms')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

  # Plot 5: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[['Rooms', 'SquareFoot', 'Location_encoded', 'Price']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.grid(False)
    plt.show()



def evaluate_model():
    if df.empty:
        messagebox.showerror("No Data", "Please load the dataset first.")
        return

    try:
        # Prepare features and target
        X = df[['Rooms', 'SquareFoot', 'Location_encoded']]
        y = df['Price']

        # Split the data (same random_state to keep consistent with training)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Compute metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        # Display results
        metrics_message = (
            f"Model Evaluation Metrics:\n\n"
            f"Mean Absolute Error (MAE): ${mae:,.2f}\n"
            f"Mean Squared Error (MSE): ${mse:,.2f}\n"
            f"Root Mean Squared Error (RMSE): ${rmse:,.2f}\n"
            f"R² Score: {r2:.2f}"
        )

        messagebox.showinfo("Model Evaluation", metrics_message)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to evaluate model: {e}")




# Create main application window
app = tk.Tk()
app.title("Residential Property Valuation")

# Load dataset button
load_button = tk.Button(app, text="Load Dataset", command=load_dataset)
load_button.grid(row=0, column=0, columnspan=2, pady=10)

# Rooms input
tk.Label(app, text="Number of Rooms:").grid(row=1, column=0, padx=10, pady=5)
rooms_entry = tk.Entry(app)
rooms_entry.grid(row=1, column=1, padx=10, pady=5)

# Square Footage input
tk.Label(app, text="Square Footage:").grid(row=2, column=0, padx=10, pady=5)
sqft_entry = tk.Entry(app)
sqft_entry.grid(row=2, column=1, padx=10, pady=5)

# Location dropdown input
tk.Label(app, text="Location:").grid(row=3, column=0, padx=10, pady=5)
location_var = tk.StringVar(app)
location_var.set("Select a location")
location_menu = tk.OptionMenu(app, location_var, "Select a location")
location_menu.grid(row=3, column=1, padx=10, pady=5)

# Predict button
predict_button = tk.Button(app, text="Predict Price", command=predict_price)
predict_button.grid(row=4, column=0, columnspan=2, pady=10)

# Result display
result_label = tk.Label(app, text="Predicted Price will appear here.")
result_label.grid(row=5, column=0, columnspan=2, pady=10)

# Add Convert to INR button
convert_button = tk.Button(app, text="Convert to INR", command=convert_to_inr)
convert_button.grid(row=7, column=0, columnspan=2, pady=10)

# Result display for INR
inr_result_label = tk.Label(app, text="Predicted Price in INR will appear here.")
inr_result_label.grid(row=8, column=0, columnspan=2, pady=10)

save_button = tk.Button(app, text="Save Prediction", command=save_prediction)
save_button.grid(row=9, column=0, columnspan=2, pady=10)

# Add button to GUI to trigger visualization
visualize_button = tk.Button(app, text="Visualize Data", command=visualize_data)
visualize_button.grid(row=10, column=0, columnspan=2, pady=10)

# Add Evaluate Model button
evaluate_button = tk.Button(app, text="Evaluate Model", command=evaluate_model)
evaluate_button.grid(row=11, column=0, columnspan=2, pady=10)

# Add Evaluate Model button to the GUI
evaluate_button = tk.Button(app, text="Evaluate Model", command=evaluate_model)
evaluate_button.grid(row=11, column=0, columnspan=2, pady=10)


# Run the application
app.mainloop()