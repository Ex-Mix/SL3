import os
import customtkinter as ctk
import numpy as np
import joblib
import pandas as pd
import logging
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from PIL import Image


# Set up logging configuration
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

# Set CustomTkinter appearance and theme
ctk.set_appearance_mode("dark")  # Options: "dark", "light", "system"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"

# Create main window
root = ctk.CTk()
root.title("Microplastic Classification")
root.geometry("800x900")

# Sample data (initially empty, populated by file selection)
sample_data = []

# Paths to model directories
ml_model_folder_path = "./model/ml"
dl_model_folder_path = "./model/dl"

# Define available ML and DL models
ml_models = ["SVC", "KNN", "LDA", "GNB", "DecisionTree", "RandomForest", "ExtraTree"]
dl_models = ["LeNet5", "AlexNet", "VGG16", "GoogLeNet", "ResNet", "Xception", "MobileNet"]

# Variables to hold the selected model names for ML and DL
selected_ml_model = ctk.StringVar(value=ml_models[0] if ml_models else "")
selected_dl_model = ctk.StringVar(value=dl_models[0] if dl_models else "")

# Define the material names corresponding to prediction labels
NameList = ['HDPE', 'LDPE', 'PET', 'PP', 'PS', 'PVC']

# Function to preprocess spectral data
def preprocess_data(data):
    data = data - np.min(data)  # Baseline correction
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).flatten().reshape(1, -1)

# Function to load data from Excel/CSV files
def load_test_data(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, header=None)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, header=None)
    else:
        return None

    if df.shape[1] > 1:
        data = df[1].values
        return preprocess_data(data), data
    else:
        raise ValueError("The selected file does not contain enough columns.")

# Function to update sample data with selected files
def select_files():
    file_paths = filedialog.askopenfilenames(title="Select Files", filetypes=(("CSV files", "*.csv"),("Excel files", "*.xlsx"), ("All files", "*.*")))
    sample_data.clear()
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        sample_data.append((file_name, "Unknown", file_path))
    update_treeview()

# Function to manually add multiple files
def add_files():
    file_paths = filedialog.askopenfilenames(title="Select Files", filetypes=(("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")))
    if file_paths:
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            sample_data.append((file_name, "Unknown", file_path))
        update_treeview()

# Function to delete the selected files
def delete_selected_files():
    selected_items = tree.selection()
    if not selected_items:
        messagebox.showwarning("Warning", "No files selected for deletion.")
        return
    
    selected_indices = sorted((tree.index(item) for item in selected_items), reverse=True)
    for index in selected_indices:
        del sample_data[index]
    
    update_treeview()

# Function to update Treeview with sample data
def update_treeview():
    for item in tree.get_children():
        tree.delete(item)
    for file_name, prediction, _ in sample_data:
        tree.insert("", "end", values=(file_name, prediction))

# Function to plot selected sample data in GUI
def plot_selected_sample_in_gui():
    selected_item = tree.selection()
    if not selected_item:
        messagebox.showwarning("Warning", "No file selected for plotting.")
        return

    selected_index = tree.index(selected_item)
    file_name, _, file_path = sample_data[selected_index]
    
    try:
        _, raw_data = load_test_data(file_path)
        
        for widget in plot_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(raw_data)
        ax.set_title(f"Spectral Data Plot - {file_name}", fontsize=10)
        ax.set_xlabel("Wavelength (cm-1)")
        ax.set_ylabel("Absorbance")

        fig.subplots_adjust(bottom=0.3)
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    except Exception as e:
        messagebox.showerror("Error", f"Error in plotting: {e}")
        logging.error(f"Error in plotting: {e}")

# Function to make predictions and update sample data
def predict_material():
    global sample_data
    predictions = []
    
    try:
        # Initialize progress bar
        progress["value"] = 0
        progress["maximum"] = len(sample_data)
        root.update_idletasks()

        if tab_control.get() == "Machine Learning Models":  # Check the selected tab by name
            model_path = Path(ml_model_folder_path) / f"model_{selected_ml_model.get()}.pkl"
            if not model_path.exists():
                raise ValueError(f"No model found for {selected_ml_model.get()}")
            
            loaded_model = joblib.load(model_path)
            
            for idx, (_, _, file_path) in enumerate(sample_data):
                X_test, _ = load_test_data(file_path)
                
                if hasattr(loaded_model, "predict_proba"):
                    proba = loaded_model.predict_proba(X_test)[0]
                    prediction = np.argmax(proba)
                    confidence_percentage = proba[prediction] * 100
                else:
                    prediction = loaded_model.predict(X_test)[0]
                    confidence_percentage = 100

                predicted_name = NameList[prediction]
                predictions.append(f"{predicted_name} ({confidence_percentage:.2f}% confidence)")
                sample_data[idx] = (sample_data[idx][0], predictions[-1], file_path)

                # Update progress bar
                progress["value"] = idx + 1
                root.update_idletasks()

        elif tab_control.get() == "Deep Learning Models":  # Check the selected tab by name
            model_path = Path(dl_model_folder_path) / f"model_{selected_dl_model.get()}.h5"
            if not model_path.exists():
                raise ValueError(f"No model found for {selected_dl_model.get()}")
            
            loaded_model = load_model(model_path)
            
            for idx, (_, _, file_path) in enumerate(sample_data):
                X_test, _ = load_test_data(file_path)
                
                proba = loaded_model.predict(X_test)[0]
                prediction = np.argmax(proba)
                confidence_percentage = proba[prediction] * 100

                predicted_name = NameList[prediction]
                predictions.append(f"{predicted_name} ({confidence_percentage:.2f}% confidence)")
                sample_data[idx] = (sample_data[idx][0], predictions[-1], file_path)

                # Update progress bar
                progress["value"] = idx + 1
                root.update_idletasks()

        # Update Treeview with predictions
        for idx, (_, _, _) in enumerate(sample_data):
            tree.item(tree.get_children()[idx], values=(sample_data[idx][0], predictions[idx]))

        # Reset progress bar after completion
        progress["value"] = 0

    except Exception as e:
        messagebox.showerror("Error", f"Error in prediction: {e}")
        logging.error(f"Error in prediction: {e}")


# Function to export results to a PDF with a progress bar
def export_results_to_pdf():
    save_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
    if not save_path:
        messagebox.showwarning("Warning", "Save path not provided. Export canceled.")
        return

    # Set up the progress bar
    progress["value"] = 0
    progress["maximum"] = len(sample_data)
    root.update_idletasks()  # Ensures the initial progress bar update

    with PdfPages(save_path) as pdf:
        for idx, (file_name, prediction, file_path) in enumerate(sample_data):
            fig, ax = plt.subplots(figsize=(8, 11))
            ax.axis('off')

            # Determine the model used based on the selected tab
            model_name = (
                selected_ml_model.get() if tab_control.get() == "Machine Learning Models" else selected_dl_model.get()
            )
            if prediction == "Unknown":
                model_name = "Not Predicted"

            header_text = (
                f"Material Classification Report\n"
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"File: {file_name}\n"
                f"Prediction: {prediction}\n"
                f"Model Used: {model_name}\n"
            )

            ax.text(0.5, 1.05, header_text, ha="center", va="top", fontsize=12, transform=ax.transAxes, linespacing=1.5)
            try:
                _, raw_data = load_test_data(file_path)
                plot_ax = fig.add_axes([0.15, 0.2, 0.7, 0.6])
                plot_ax.plot(raw_data)
                plot_ax.set_title(f"Spectral Data Plot - {file_name}")
                plot_ax.set_xlabel("Wavelength (cm-1)")
                plot_ax.set_ylabel("Absorbance")
            except Exception as e:
                logging.error(f"Error loading data for {file_name}: {e}")
                ax.text(0.5, 0.5, f"Error loading data for {file_name}: {e}", ha="center", va="center", fontsize=10)

            pdf.savefig(fig)
            plt.close(fig)

            # Update the progress bar for each file processed
            progress["value"] = idx + 1
            root.update_idletasks()  # Force update the progress bar display

        messagebox.showinfo("Info", "Results with plots exported to PDF successfully.")
    
    # Reset the progress bar once completed
    progress["value"] = 0


#GUI LAYOUT

# Set row and column weights for main root window to make it responsive
root.grid_rowconfigure(0, weight=0)  # File control buttons, no expansion needed
root.grid_rowconfigure(1, weight=0)  # ML/DL model selection tabs
root.grid_rowconfigure(2, weight=0)  # Prediction and control buttons
root.grid_rowconfigure(3, weight=1)  # Plot area, main expandable row
root.grid_columnconfigure(0, weight=1)  # Left-side controls
root.grid_columnconfigure(1, weight=1)  # Right-side result section

# File Control Buttons
file_control_frame = ctk.CTkFrame(root)
file_control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

select_files_button = ctk.CTkButton(file_control_frame, text="Select Files", command=select_files)
select_files_button.pack(side="left", padx=(0, 5))
add_file_button = ctk.CTkButton(file_control_frame, text="Add Files", command=add_files)
add_file_button.pack(side="left", padx=(0, 5))
delete_file_button = ctk.CTkButton(file_control_frame, text="Delete Files", command=delete_selected_files)
delete_file_button.pack(side="left")

# Tabs for ML and DL Model Selection
tab_control = ctk.CTkTabview(root)
tab_control.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

# Machine Learning Model Selection
ml_tab = tab_control.add("Machine Learning Models")
ml_model_frame = ctk.CTkFrame(ml_tab)
ml_model_frame.pack(fill="both", expand=True, padx=10, pady=5)
ml_model_label = ctk.CTkLabel(ml_model_frame, text="Select ML Model", font=("Arial", 14, "bold"))

ml_model_label.pack(anchor="w")
for model in ml_models:
    ctk.CTkRadioButton(ml_model_frame, text=model, variable=selected_ml_model, value=model).pack(anchor="w")

# Deep Learning Model Selection
dl_tab = tab_control.add("Deep Learning Models")
dl_model_frame = ctk.CTkFrame(dl_tab)
dl_model_frame.pack(fill="both", expand=True, padx=10, pady=5)
dl_model_label = ctk.CTkLabel(dl_model_frame, text="Select DL Model", font=("Arial", 14, "bold"))

dl_model_label.pack(anchor="w")
for model in dl_models:
    ctk.CTkRadioButton(dl_model_frame, text=model, variable=selected_dl_model, value=model).pack(anchor="w")

# Prediction and Control Buttons
classify_frame = ctk.CTkFrame(root)
classify_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
plot_button = ctk.CTkButton(classify_frame, text="Plot Spectral Data", command=plot_selected_sample_in_gui)
plot_button.grid(row=0, column=0, pady=5)
predict_button = ctk.CTkButton(classify_frame, text="Predict Material", command=predict_material)
predict_button.grid(row=1, column=0, pady=5)
export_button = ctk.CTkButton(classify_frame, text="Export a Report (PDF)", command=export_results_to_pdf)
export_button.grid(row=2, column=0, pady=5)

# Results Section with Scrollbar and Progress Bar
result_frame = ctk.CTkFrame(root)
result_frame.grid(row=0, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")
result_frame.grid_rowconfigure(0, weight=1)  # Allows Treeview to expand vertically
result_frame.grid_columnconfigure(0, weight=1)  # Allows Treeview to expand horizontally

# Treeview and Scrollbar Frame
tree_scroll_frame = ctk.CTkFrame(result_frame)
tree_scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)

# Treeview for displaying files and predictions
tree = ttk.Treeview(tree_scroll_frame, columns=("File Name", "Prediction"), show="headings", height=15)
tree.heading("File Name", text="File Name")
tree.heading("Prediction", text="Prediction")
tree.pack(side="left", fill="both", expand=True)

# Scrollbar aligned with Treeview
scrollbar = ttk.Scrollbar(tree_scroll_frame, orient="vertical", command=tree.yview)
scrollbar.pack(side="right", fill="y")
tree.configure(yscrollcommand=scrollbar.set)

# Export Progress Bar at the Bottom of result_frame
progress_label = ctk.CTkLabel(result_frame, text="Progress Bar")
progress_label.pack(pady=(10, 5), side="top")
progress_frame = ctk.CTkFrame(result_frame)
progress_frame.pack(fill="x", pady=10)

progress_label = ctk.CTkLabel(progress_frame, text="Progress:")
progress_label.pack(side="left", padx=10)
progress = ttk.Progressbar(progress_frame, orient="horizontal", length=400, mode="determinate")
progress.pack(side="left", fill="x", expand=True, padx=10)


# Plot Area Frame
plot_frame = ctk.CTkFrame(root)
plot_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
plot_frame.grid_rowconfigure(0, weight=1)
plot_frame.grid_columnconfigure(0, weight=1)

# Run the main loop
root.mainloop()

